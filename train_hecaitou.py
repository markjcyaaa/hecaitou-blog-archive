#!/usr/bin/env python3
"""
和菜头风格 LoRA 微调训练脚本 v3.0
==================================
基于 Unsloth + Qwen3.5 + bf16 LoRA，使用和菜头 1062 篇文章进行风格微调。

v3.0 更新（2026-03-13）：
  - 切换为 prompt/completion 格式 + completion-only loss
    （只在文章输出部分计算 loss，不学习生成概要/指令）
  - 默认使用 Qwen3.5-2B + bf16 LoRA（Unsloth 官方推荐，不用 QLoRA）
  - max_seq_len 默认 4096（8GB 显存可用，覆盖 95%+ 样本）
  - 支持增强 prompt 格式（带概要+意向）
  - 自动检测数据格式（prompt/completion 或旧版 messages）

v2.0 修复记录（2026-03-11）：
  - 修复 formatting_func / dataset_text_field 兼容性问题
  - 修复 Unsloth SFTTrainer 对 ChatML messages 格式的解析错误
  - 添加 RTX 5060 (8GB) 本地训练支持

使用环境：
  - 本地 NVIDIA GPU 8GB (RTX 5060 等) → Qwen3.5-2B bf16 LoRA（推荐）
  - 本地 NVIDIA GPU 10GB+ → Qwen3.5-4B bf16 LoRA
  - Google Colab (免费 T4 GPU, 16GB) → Qwen3.5-4B bf16 LoRA
  - 本地 NVIDIA GPU 22GB+ (RTX 4090 等) → Qwen3.5-9B bf16 LoRA

用法：
  # 最简用法（自动检测 GPU 并选择最优配置）
  python train_hecaitou.py

  # RTX 5060 推荐配置
  python train_hecaitou.py --model Qwen/Qwen3.5-2B --max-seq-len 4096

  # 指定数据目录（使用 v3 增强数据）
  python train_hecaitou.py --data-dir ./training_data_v3

  # 只导出 GGUF（已有训练好的模型时）
  python train_hecaitou.py --export-only --model-dir ./hecaitou_output/lora_adapter

  # 从检查点恢复训练
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
    print("和菜头风格 LoRA 微调训练脚本 v3.0")
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
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / 1024**3
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
    """
    根据 GPU 显存自动选择最优训练配置。
    
    核心原则（基于 Unsloth 官方文档 2026-03）：
    - Qwen3.5 不推荐 QLoRA (4-bit)，应使用 bf16 LoRA
    - bf16 LoRA VRAM: 0.8B≈3GB, 2B≈5GB, 4B≈10GB, 9B≈22GB
    - 优先保证 max_seq_len 覆盖全部训练文本
    - 完整文本 > 截断文本（风格学习的关键）
    """
    mem = gpu_info["gpu_mem_gb"]

    if user_model:
        model = user_model
    elif mem >= 22:
        model = "Qwen/Qwen3.5-9B"
    elif mem >= 10:
        model = "Qwen/Qwen3.5-4B"
    elif mem >= 5:
        model = "Qwen/Qwen3.5-2B"   # 8GB 显存推荐
    else:
        model = "Qwen/Qwen3.5-0.8B"

    # LoRA rank
    is_9b = "9B" in model or "9b" in model
    is_4b = "4B" in model or "4b" in model
    is_2b = "2B" in model or "2b" in model

    if user_lora_r:
        lora_r = user_lora_r
    elif is_9b:
        lora_r = 32
    elif is_4b:
        lora_r = 32
    elif is_2b:
        lora_r = 32  # 2B 模型显存充裕，用大 r 提升学习能力
    else:
        lora_r = 16

    # max_seq_len：尽可能覆盖训练数据
    # 训练数据分布：95.4% ≤4096 tokens, 98.6% ≤5120
    if user_seq_len:
        max_seq_len = user_seq_len
    elif is_2b and mem >= 7:
        max_seq_len = 4096  # 2B + 8GB → 4096 覆盖 95%
    elif is_2b:
        max_seq_len = 2048
    elif is_4b and mem >= 14:
        max_seq_len = 4096
    elif is_4b:
        max_seq_len = 2048
    elif is_9b:
        max_seq_len = 2048
    else:
        max_seq_len = 4096  # 0.8B 可以开很大

    print(f"\n[自动配置] 模型: {model}")
    print(f"[自动配置] 训练模式: bf16 LoRA（Qwen3.5 官方推荐）")
    print(f"[自动配置] LoRA r: {lora_r}")
    print(f"[自动配置] 最大序列长度: {max_seq_len}")
    print(f"[自动配置] 显存预估: {'充足' if mem >= 12 else '可用（已优化参数）'}")

    return model, lora_r, max_seq_len


# ============================================================
# 数据加载
# ============================================================

def detect_data_format(data_dir: str) -> str:
    """自动检测数据格式：prompt/completion 或 messages。"""
    train_file = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(train_file):
        return "unknown"
    
    with open(train_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
    
    try:
        data = json.loads(first_line)
        if "prompt" in data and "completion" in data:
            return "prompt_completion"
        elif "messages" in data:
            return "messages"
        elif "text" in data:
            return "text"
        else:
            return "unknown"
    except:
        return "unknown"


def load_data(data_dir: str):
    """加载训练数据，自动检测格式。"""
    from datasets import load_dataset

    train_file = os.path.join(data_dir, "train.jsonl")
    eval_file = os.path.join(data_dir, "eval.jsonl")

    if not os.path.exists(train_file):
        print(f"\n[错误] 未找到训练数据: {train_file}")
        print("  请先运行: python prepare_training_data.py")
        sys.exit(1)

    data_format = detect_data_format(data_dir)
    print(f"\n[数据] 检测到格式: {data_format}")

    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = None
    if os.path.exists(eval_file):
        eval_dataset = load_dataset("json", data_files=eval_file, split="train")

    print(f"[数据] 训练集: {len(train_dataset)} 条")
    if eval_dataset:
        print(f"[数据] 验证集: {len(eval_dataset)} 条")

    return train_dataset, eval_dataset, data_format


def convert_messages_to_text(dataset, tokenizer):
    """
    将旧版 ChatML messages 格式转为纯文本（兼容 v2 数据）。
    """
    def convert_fn(example):
        messages = example["messages"]
        parts = []
        for msg in messages:
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
    print(f"  已转换 {len(converted)} 条 messages → 纯文本")
    return converted


# ============================================================
# 模型加载
# ============================================================

def load_model(model_name: str, max_seq_length: int, lora_r: int):
    """加载模型并配置 bf16 LoRA。"""
    import torch
    from unsloth import FastLanguageModel

    print(f"\n[模型] 加载 {model_name}（bf16 LoRA）...")
    print(f"  这一步需要下载模型文件，首次可能较慢...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,       # 不使用 QLoRA
        load_in_16bit=True,       # bf16 LoRA
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

def run_training(model, tokenizer, train_dataset, eval_dataset, data_format, args, gpu_info):
    """执行训练。"""
    import torch
    from trl import SFTTrainer, SFTConfig

    use_bf16 = gpu_info["bf16"]

    print(f"\n[训练] 配置:")
    print(f"  模型精度: {'bf16' if use_bf16 else 'float32 (Unsloth 自动管理)'}")
    print(f"  数据格式: {data_format}")
    print(f"  轮次 (epochs): {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  等效 batch size: {args.batch_size * args.grad_accum}")
    print(f"  最大序列长度: {args.max_seq_len}")

    # 根据数据格式配置 SFTConfig
    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": 20,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "fp16": False,
        "bf16": use_bf16,
        "seed": 42,
        "report_to": "none",
        "max_length": args.max_seq_len,
    }

    if data_format == "prompt_completion":
        # prompt/completion 格式：TRL 自动做 completion-only loss
        # 无需设置 dataset_text_field
        print(f"  Loss 模式: completion-only（只在文章部分计算 loss）")
    elif data_format == "messages":
        # messages 格式：需要设置 assistant_only_loss
        sft_kwargs["dataset_text_field"] = "text"
        print(f"  Loss 模式: 全序列 loss（旧版 messages 格式）")
    else:
        sft_kwargs["dataset_text_field"] = "text"
        print(f"  Loss 模式: 全序列 loss")

    sft_config = SFTConfig(**sft_kwargs)

    if eval_dataset:
        sft_config.eval_strategy = "steps"
        sft_config.eval_steps = 100

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    total_steps = len(train_dataset) * args.epochs // (args.batch_size * args.grad_accum)
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
    """保存 LoRA 适配器。"""
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
    """保存合并后的 16-bit 模型。"""
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
        description="和菜头风格 LoRA 微调训练 v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 最简用法（自动检测 GPU 并选择最优配置）
  python train_hecaitou.py

  # RTX 5060 (8GB) 推荐配置
  python train_hecaitou.py --model Qwen/Qwen3.5-2B --max-seq-len 4096

  # 使用 v3 增强数据集
  python train_hecaitou.py --data-dir ./training_data_v3

  # 从中断处恢复
  python train_hecaitou.py --resume

  # 只导出 GGUF
  python train_hecaitou.py --export-only --model-dir ./hecaitou_output/lora_adapter
        """,
    )
    parser.add_argument("--data-dir", default="training_data", help="训练数据目录（默认 training_data/）")
    parser.add_argument("--output-dir", default="hecaitou_output", help="输出目录（默认 hecaitou_output/）")
    parser.add_argument("--model", default=None, help="基座模型（默认自动选择：8GB→2B, 10GB+→4B, 22GB+→9B）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次（默认 3）")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率（默认 2e-4）")
    parser.add_argument("--batch-size", type=int, default=1, help="批大小（默认 1，不建议改）")
    parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数（默认 4，等效 batch=4）")
    parser.add_argument("--max-seq-len", type=int, default=None, help="最大序列长度（默认自动：8GB→4096, 其他→2048）")
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA 秩（默认 32）")
    parser.add_argument("--quant", default="q4_k_m", help="GGUF 量化方法（默认 q4_k_m）")
    parser.add_argument("--no-export", action="store_true", help="训练完不导出 GGUF（只保存 LoRA）")
    parser.add_argument("--export-only", action="store_true", help="只导出 GGUF，不训练")
    parser.add_argument("--model-dir", default=None, help="已训练模型目录（配合 --export-only）")
    parser.add_argument("--resume", action="store_true", help="从最近的检查点恢复训练")
    parser.add_argument("--cache-dir", default=None, help="模型缓存目录（默认 C 盘，如 C 盘空间不足可设为 D:\\hf_cache）")
    args = parser.parse_args()

    # ---- 设置缓存目录（解决 C 盘空间不足问题） ----
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")
        print(f"[缓存] 模型缓存目录: {args.cache_dir}")

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
            load_in_4bit=False,
            load_in_16bit=True,
        )
        export_gguf(model, tokenizer, args.output_dir, args.quant)
        return

    # ---- 自动选择配置 ----
    model_name, lora_r, max_seq_len = auto_select_config(
        gpu_info, args.model, args.lora_r, args.max_seq_len
    )
    args.max_seq_len = max_seq_len

    # ---- 加载数据 ----
    train_dataset, eval_dataset, data_format = load_data(args.data_dir)

    # ---- 加载模型 ----
    model, tokenizer = load_model(model_name, max_seq_len, lora_r)

    # ---- 数据预处理（根据格式） ----
    if data_format == "messages":
        # 旧版 messages 格式：需要转为纯文本
        print("\n[数据] 检测到旧版 messages 格式，转换为纯文本...")
        train_dataset = convert_messages_to_text(train_dataset, tokenizer)
        if eval_dataset:
            eval_dataset = convert_messages_to_text(eval_dataset, tokenizer)
        data_format = "text"  # 转换后变成 text 格式
    elif data_format == "prompt_completion":
        # 新版 prompt/completion 格式：TRL 原生支持，无需转换
        print("\n[数据] prompt/completion 格式，启用 completion-only loss")
    
    # ---- 训练 ----
    model, tokenizer = run_training(
        model, tokenizer, train_dataset, eval_dataset, data_format, args, gpu_info
    )

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
    print(f"\n下一步（Windows）：")
    print(f"  1. 合并 LoRA 为完整模型:")
    print(f'     python -c "')
    print(f'     import torch')
    print(f'     from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'     from peft import PeftModel')
    print(f'     tokenizer = AutoTokenizer.from_pretrained(\\"{args.output_dir}/lora_adapter\\", local_files_only=True)')
    print(f'     base_model = AutoModelForCausalLM.from_pretrained(\\"{model_name}\\", torch_dtype=torch.float16, device_map=\\"cpu\\", local_files_only=True)')
    print(f'     model = PeftModel.from_pretrained(base_model, \\"{args.output_dir}/lora_adapter\\", device_map=\\"cpu\\")')
    print(f'     model = model.merge_and_unload()')
    print(f'     model.save_pretrained(\\"D:/hecaitou_merged\\")')
    print(f'     tokenizer.save_pretrained(\\"D:/hecaitou_merged\\")')
    print(f'     "')
    print(f"  2. 用 convert_hf_to_gguf.py 转为 GGUF")
    print(f"  3. 导入 Ollama 测试")
    if gguf_dir:
        print(f"\n  或直接使用 GGUF:")
        print(f"  ollama create hecaitou-writer -f Modelfile")
        print(f"  ollama run hecaitou-writer '写一篇关于...'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
