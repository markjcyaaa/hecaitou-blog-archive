#!/usr/bin/env python3
"""
和菜头风格 LoRA 微调训练脚本 v2.0
==================================
基于 Unsloth + Qwen3.5 + QLoRA，使用和菜头 1062 篇文章进行风格微调。

修复记录（v2.0 - 2026-03-11）：
  - 修复 formatting_func / dataset_text_field 兼容性问题
  - 修复 Unsloth SFTTrainer 对 ChatML messages 格式的解析错误
  - 修复 T4/RTX 50xx 上 float16/bf16 自动检测失败的问题
  - 添加 RTX 5060 (8GB) 本地训练支持（自动选择 4B 模型）
  - 添加 Conda 环境安装指引

使用环境：
  - Google Colab (免费 T4 GPU, 16GB 显存) → Qwen3.5-4B (推荐) 或 9B
  - 本地 NVIDIA GPU 8GB+ (RTX 5060 等) → Qwen3.5-4B
  - 本地 NVIDIA GPU 12GB+ (RTX 4090 等) → Qwen3.5-9B

用法：
  # Windows 本地训练（RTX 5060, 8GB 显存）- 推荐
  python train_hecaitou.py

  # 本地训练指定 4B 模型（8GB 显存卡推荐）
  python train_hecaitou.py --model Qwen/Qwen3.5-4B

  # 本地训练指定 9B 模型（12GB+ 显存）
  python train_hecaitou.py --model Qwen/Qwen3.5-9B --lora-r 32

  # 指定数据目录
  python train_hecaitou.py --data-dir ./training_data

  # 只导出 GGUF（已有训练好的模型时）
  python train_hecaitou.py --export-only --model-dir ./hecaitou_output/checkpoint-xxx

  # 从检查点恢复训练（如果训练中途中断）
  python train_hecaitou.py --resume

依赖安装（Windows Conda 方式 - 推荐）：
  conda create --name unsloth_env python==3.12 -y
  conda activate unsloth_env
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip install unsloth
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path


# ============================================================
# 环境检查
# ============================================================

def check_environment():
    """检查运行环境，返回 GPU 信息。"""
    print("=" * 60)
    print("和菜头风格 LoRA 微调训练脚本 v2.0")
    print("=" * 60)

    # 检查 GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n[错误] 未检测到 NVIDIA GPU 或 CUDA 不可用。")
            print("  请确保：")
            print("  1. 安装了 NVIDIA 显卡驱动")
            print("  2. 安装了正确版本的 PyTorch（带 CUDA）")
            print("  3. 如果用 Conda：conda activate unsloth_env")
            print("  运行 nvidia-smi 确认 GPU 状态")
            return None
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        compute_cap = torch.cuda.get_device_capability(0)
        bf16_ok = torch.cuda.is_bf16_supported()
        print(f"\n[环境] GPU: {gpu_name}")
        print(f"[环境] 显存: {gpu_mem_gb:.1f} GB")
        print(f"[环境] Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        print(f"[环境] BF16 支持: {'是' if bf16_ok else '否'}")
        print(f"[环境] PyTorch: {torch.__version__}")
        print(f"[环境] CUDA: {torch.version.cuda}")
        return {
            "gpu_name": gpu_name,
            "gpu_mem_gb": gpu_mem_gb,
            "bf16": bf16_ok,
            "compute_cap": compute_cap,
        }
    except ImportError:
        print("\n[错误] 未安装 PyTorch。")
        print("  Conda 用户: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        return None


def auto_select_config(gpu_info, user_model=None, user_lora_r=None, user_seq_len=None):
    """根据 GPU 显存自动选择最优训练配置。"""
    mem = gpu_info["gpu_mem_gb"]

    if user_model:
        model = user_model
    elif mem >= 14:
        model = "Qwen/Qwen3.5-9B"
    elif mem >= 6:
        model = "Qwen/Qwen3.5-4B"
    else:
        model = "Qwen/Qwen3.5-2B"

    # 根据模型大小和显存选择参数
    is_9b = "9B" in model or "9b" in model
    is_4b = "4B" in model or "4b" in model

    if user_lora_r:
        lora_r = user_lora_r
    elif is_9b:
        lora_r = 32 if mem >= 14 else 16
    elif is_4b:
        lora_r = 16 if mem < 10 else 32
    else:
        lora_r = 16

    if user_seq_len:
        max_seq_len = user_seq_len
    elif mem >= 14:
        max_seq_len = 2048
    elif mem >= 8:
        max_seq_len = 1024
    else:
        max_seq_len = 512

    print(f"\n[自动配置] 模型: {model}")
    print(f"[自动配置] LoRA r: {lora_r}")
    print(f"[自动配置] 最大序列长度: {max_seq_len}")
    print(f"[自动配置] 显存余量: {'充足' if mem >= 12 else '紧张，已优化参数'}")

    return model, lora_r, max_seq_len


# ============================================================
# 数据加载
# ============================================================

def load_data(data_dir: str):
    """加载训练数据。"""
    from datasets import load_dataset

    train_file = os.path.join(data_dir, "train.jsonl")
    eval_file = os.path.join(data_dir, "eval.jsonl")

    if not os.path.exists(train_file):
        print(f"\n[错误] 未找到训练数据: {train_file}")
        print("  请先运行: python prepare_training_data.py")
        sys.exit(1)

    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = None
    if os.path.exists(eval_file):
        eval_dataset = load_dataset("json", data_files=eval_file, split="train")

    print(f"\n[数据] 训练集: {len(train_dataset)} 条")
    if eval_dataset:
        print(f"[数据] 验证集: {len(eval_dataset)} 条")

    return train_dataset, eval_dataset


def convert_dataset_to_text(dataset, tokenizer):
    """
    将 ChatML messages 格式的数据集预转换为纯文本。
    
    这是关键修复：Unsloth 的 SFTTrainer 在处理 messages 格式时存在兼容性问题
    （formatting_func 返回空列表、JSON 解析错误等）。
    预转换为纯文本列 "text" 可以完全绕过这些问题。
    """
    def convert_fn(example):
        messages = example["messages"]
        parts = []
        for msg in messages:
            # 处理 msg 可能是 dict 或已序列化的 JSON string
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        text = "\n".join(parts)
        return {"text": text}

    converted = dataset.map(convert_fn, remove_columns=dataset.column_names)
    print(f"  已转换 {len(converted)} 条为纯文本格式")

    # 验证
    sample = converted[0]["text"]
    if "<|im_start|>" not in sample:
        print("  [警告] 转换后的文本不含 ChatML 标记，请检查数据格式！")
    else:
        print(f"  预览（前 150 字）: {sample[:150]}...")

    return converted


# ============================================================
# 模型加载
# ============================================================

def load_model(model_name: str, max_seq_length: int, lora_r: int):
    """加载模型并配置 LoRA。"""
    import torch
    from unsloth import FastLanguageModel

    print(f"\n[模型] 加载 {model_name}（4-bit 量化）...")
    print(f"  这一步需要下载模型文件，首次可能较慢...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    print(f"\n[模型] 配置 LoRA（r={lora_r}, alpha={lora_r}）...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )

    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"\n[模型] 总参数: {total_params / 1e9:.2f}B")
    print(f"[模型] 可训练: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"[模型] 当前显存占用: {mem_used:.1f} GB")

    return model, tokenizer


# ============================================================
# 训练
# ============================================================

def run_training(model, tokenizer, train_text, eval_text, args, gpu_info):
    """执行训练。"""
    import torch
    from trl import SFTTrainer, SFTConfig

    # 精度设置
    # RTX 5060 (Blackwell, sm_120) 和 RTX 40xx (Ada, sm_89) 支持 bf16
    # Tesla T4 (Turing, sm_75) 不支持 bf16，但 Unsloth 会自动切到 float32
    # 这里显式设为 False 避免冲突，让 Unsloth 自行决定
    use_bf16 = gpu_info["bf16"]
    use_fp16 = False  # 让 Unsloth 处理精度降级

    # 如果 Unsloth 不支持 fp16（如某些 Qwen3.5 模型），全部设为 False
    # Unsloth 会自动切换到 float32
    print(f"\n[训练] 配置:")
    print(f"  模型精度: {'bf16' if use_bf16 else 'float32 (Unsloth 自动管理)'}")
    print(f"  轮次 (epochs): {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  等效 batch size: {args.batch_size * args.grad_accum}")
    print(f"  最大序列长度: {args.max_seq_len}")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",  # 使用预转换的纯文本列
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=42,
        report_to="none",
    )

    if eval_text:
        sft_config.eval_strategy = "steps"
        sft_config.eval_steps = 100

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text,
        eval_dataset=eval_text,
        args=sft_config,
    )

    total_steps = len(train_text) * args.epochs // (args.batch_size * args.grad_accum)
    est_min_low = total_steps * 4 / 60
    est_min_high = total_steps * 6 / 60
    print(f"\n  预计总步数: ~{total_steps}")
    print(f"  预计用时: {est_min_low:.0f} - {est_min_high:.0f} 分钟")
    print(f"\n{'=' * 50}")
    print("开始训练... (Ctrl+C 可中断，下次用 --resume 恢复)")
    print(f"{'=' * 50}\n")

    start_time = time.time()

    if args.resume:
        print("[恢复] 从最近的检查点恢复训练...")
        stats = trainer.train(resume_from_checkpoint=True)
    else:
        stats = trainer.train()

    elapsed = (time.time() - start_time) / 60

    print(f"\n{'=' * 50}")
    print(f"训练完成！")
    print(f"  总步数: {stats.global_step}")
    print(f"  最终 loss: {stats.training_loss:.4f}")
    print(f"  实际用时: {elapsed:.0f} 分钟")
    print(f"{'=' * 50}")

    return model, tokenizer


# ============================================================
# 导出
# ============================================================

def export_gguf(model, tokenizer, output_dir: str, quant_method: str = "q4_k_m"):
    """导出为 GGUF 格式。"""
    gguf_dir = os.path.join(output_dir, "gguf")

    print(f"\n[导出] 正在导出 GGUF（{quant_method}）...")
    print("  这一步需要 10-15 分钟，请耐心等待。")

    try:
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant_method)
    except Exception as e:
        print(f"\n  [警告] GGUF 导出失败: {e}")
        print("  可能原因：Windows 不支持 GGUF 导出，或缺少 llama.cpp。")
        print("  替代方案：先保存 LoRA 适配器，再手动转换。")
        print("  参考：https://github.com/ggml-org/llama.cpp")
        return None

    gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
    if gguf_files:
        for f in gguf_files:
            size_mb = os.path.getsize(f) / 1024 / 1024
            print(f"  {os.path.basename(f)} ({size_mb:.0f} MB)")
        return gguf_dir
    else:
        print("  [警告] 未找到 GGUF 文件，请检查导出过程。")
        return None


def export_lora(model, tokenizer, output_dir: str):
    """保存 LoRA 适配器（体积小，方便传输和继续训练）。"""
    lora_dir = os.path.join(output_dir, "lora_adapter")
    print(f"\n[导出] 保存 LoRA 适配器到 {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    total_size = sum(
        os.path.getsize(os.path.join(lora_dir, f))
        for f in os.listdir(lora_dir) if os.path.isfile(os.path.join(lora_dir, f))
    )
    print(f"  LoRA 适配器已保存 ({total_size / 1024 / 1024:.0f} MB)")
    return lora_dir


def export_merged_16bit(model, tokenizer, output_dir: str):
    """保存合并后的 16-bit 模型（可用于 vLLM 等推理框架）。"""
    merged_dir = os.path.join(output_dir, "merged_16bit")
    print(f"\n[导出] 保存 16-bit 合并模型到 {merged_dir}")
    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"  16-bit 合并模型已保存")
        return merged_dir
    except Exception as e:
        print(f"  [跳过] 16-bit 导出失败: {e}")
        return None


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="和菜头风格 LoRA 微调训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 最简用法（自动检测 GPU 并选择最优配置）
  python train_hecaitou.py

  # RTX 5060 (8GB) 本地训练
  python train_hecaitou.py --model Qwen/Qwen3.5-4B

  # RTX 4090 (24GB) 本地训练
  python train_hecaitou.py --model Qwen/Qwen3.5-9B --lora-r 32 --max-seq-len 2048

  # 从中断处恢复
  python train_hecaitou.py --resume

  # 只导出 GGUF
  python train_hecaitou.py --export-only --model-dir ./hecaitou_output/checkpoint-xxx
        """,
    )
    parser.add_argument("--data-dir", default="training_data", help="训练数据目录（默认 training_data/）")
    parser.add_argument("--output-dir", default="hecaitou_output", help="输出目录（默认 hecaitou_output/）")
    parser.add_argument("--model", default=None, help="基座模型（默认自动选择：8GB→4B, 12GB+→9B）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次（默认 3）")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率（默认 2e-4）")
    parser.add_argument("--batch-size", type=int, default=1, help="批大小（默认 1，不建议改）")
    parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数（默认 4，等效 batch=4）")
    parser.add_argument("--max-seq-len", type=int, default=None, help="最大序列长度（默认自动：8GB→1024, 12GB+→2048）")
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA 秩（默认自动：8GB→16, 12GB+→32）")
    parser.add_argument("--quant", default="q4_k_m", help="GGUF 量化方法（默认 q4_k_m）")
    parser.add_argument("--no-export", action="store_true", help="训练完不导出 GGUF（只保存 LoRA）")
    parser.add_argument("--export-only", action="store_true", help="只导出 GGUF，不训练")
    parser.add_argument("--model-dir", default=None, help="已训练模型目录（配合 --export-only）")
    parser.add_argument("--resume", action="store_true", help="从最近的检查点恢复训练")
    args = parser.parse_args()

    # ---- 环境检查 ----
    gpu_info = check_environment()
    if gpu_info is None:
        sys.exit(1)

    # ---- 检查 unsloth ----
    try:
        import unsloth
        ver = getattr(unsloth, "__version__", "已安装")
        print(f"[环境] Unsloth: {ver}")
    except ImportError:
        print("\n[错误] 未安装 Unsloth。")
        print("  推荐安装方式（Windows Conda）：")
        print("    conda create --name unsloth_env python==3.12 -y")
        print("    conda activate unsloth_env")
        print("    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print("    pip install unsloth")
        sys.exit(1)

    # ---- 检查 trl ----
    try:
        import trl
        print(f"[环境] TRL: {trl.__version__}")
    except ImportError:
        print("\n[错误] 未安装 TRL。请运行: pip install trl datasets")
        sys.exit(1)

    # ---- 导出模式 ----
    if args.export_only:
        if not args.model_dir:
            print("[错误] --export-only 需要配合 --model-dir 指定已训练模型路径")
            sys.exit(1)
        from unsloth import FastLanguageModel
        seq_len = args.max_seq_len or 2048
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_dir,
            max_seq_length=seq_len,
            load_in_4bit=True,
        )
        export_gguf(model, tokenizer, args.output_dir, args.quant)
        return

    # ---- 自动选择配置 ----
    model_name, lora_r, max_seq_len = auto_select_config(
        gpu_info, args.model, args.lora_r, args.max_seq_len
    )
    args.max_seq_len = max_seq_len

    # ---- 加载数据 ----
    train_dataset, eval_dataset = load_data(args.data_dir)

    # ---- 加载模型 ----
    model, tokenizer = load_model(model_name, max_seq_len, lora_r)

    # ---- 预转换数据为纯文本 ----
    # 这一步解决了 Unsloth SFTTrainer 对 messages 格式的各种兼容性 bug
    print("\n[数据] 预转换为 ChatML 纯文本...")
    train_text = convert_dataset_to_text(train_dataset, tokenizer)
    eval_text = convert_dataset_to_text(eval_dataset, tokenizer) if eval_dataset else None

    # ---- 训练 ----
    model, tokenizer = run_training(model, tokenizer, train_text, eval_text, args, gpu_info)

    # ---- 导出 ----
    export_lora(model, tokenizer, args.output_dir)

    if not args.no_export:
        gguf_dir = export_gguf(model, tokenizer, args.output_dir, args.quant)
    else:
        gguf_dir = None

    # ---- 完成 ----
    print(f"\n{'=' * 60}")
    print("全部完成！")
    print(f"\n  LoRA 适配器: {args.output_dir}/lora_adapter/")
    if gguf_dir:
        print(f"  GGUF 模型: {args.output_dir}/gguf/*.gguf")
    print(f"\n下一步：")
    if gguf_dir:
        print(f"  1. 在 GGUF 文件同目录创建 Modelfile（见下方）")
        print(f"  2. ollama create hecaitou-writer -f Modelfile")
        print(f"  3. ollama run hecaitou-writer '写一篇关于...'")
        print(f"\n  Modelfile 内容：")
        print(f'    FROM ./{args.quant.upper()}.gguf')
        print(f'    PARAMETER temperature 0.7')
        print(f'    PARAMETER num_ctx 16384')
        print(f'    SYSTEM "你是和菜头，运营公众号「槽边往事」。"')
    else:
        print(f"  GGUF 导出失败/跳过。可手动导出：")
        print(f"  python train_hecaitou.py --export-only --model-dir {args.output_dir}/lora_adapter")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
