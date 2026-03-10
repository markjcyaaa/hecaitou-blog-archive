#!/usr/bin/env python3
"""
和菜头风格 LoRA 微调训练脚本
=============================
基于 Unsloth + Qwen3.5-9B + QLoRA，使用和菜头 1062 篇文章进行风格微调。

使用环境：
  - Google Colab (免费 T4 GPU)
  - 或本地 NVIDIA GPU (12GB+ 显存)

用法：
  # 本地训练（需要先生成训练数据）
  python train_hecaitou.py

  # 指定数据目录
  python train_hecaitou.py --data-dir ./training_data

  # 调整训练参数
  python train_hecaitou.py --epochs 2 --lora-r 64 --max-seq-len 4096

  # 只导出 GGUF（已有训练好的模型时）
  python train_hecaitou.py --export-only --model-dir ./hecaitou_output/checkpoint-xxx

依赖：
  pip install unsloth unsloth_zoo trl datasets
"""

import argparse
import glob
import os
import sys
import shutil
from pathlib import Path


def check_environment():
    """检查运行环境。"""
    print("=" * 60)
    print("和菜头风格 LoRA 微调训练脚本")
    print("=" * 60)

    # 检查 GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n[错误] 未检测到 NVIDIA GPU。")
            print("  本脚本需要 NVIDIA GPU（至少 12GB 显存）。")
            print("  如果你没有 GPU，请使用 Google Colab 免费训练。")
            print("  详细步骤请看 FINETUNE_GUIDE.md 的 Step 3。")
            return False
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"\n[环境] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    except ImportError:
        print("\n[错误] 未安装 PyTorch。请运行: pip install torch")
        return False

    # 检查 unsloth
    try:
        import unsloth
        print(f"[环境] Unsloth: {unsloth.__version__ if hasattr(unsloth, '__version__') else '已安装'}")
    except ImportError:
        print("\n[错误] 未安装 Unsloth。请运行:")
        print("  pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo")
        return False

    # 检查 trl
    try:
        import trl
        print(f"[环境] TRL: {trl.__version__}")
    except ImportError:
        print("\n[错误] 未安装 TRL。请运行: pip install trl datasets")
        return False

    return True


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


def load_model(model_name: str, max_seq_length: int, lora_r: int):
    """加载模型并配置 LoRA。"""
    from unsloth import FastLanguageModel

    print(f"\n[模型] 加载 {model_name}（4-bit 量化）...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    print(f"[模型] 配置 LoRA（r={lora_r}）...")
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
    import torch
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params / 1e9:.2f}B")
    print(f"[模型] 可训练: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"[模型] 显存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    return model, tokenizer


def train(model, tokenizer, train_dataset, eval_dataset, args):
    """执行训练。"""
    import torch
    from trl import SFTTrainer, SFTConfig

    print(f"\n[训练] 配置:")
    print(f"  轮次 (epochs): {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  等效 batch size: {args.batch_size * args.grad_accum}")
    print(f"  最大序列长度: {args.max_seq_len}")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_len,
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
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=42,
        report_to="none",
    )

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
    print(f"  预计总步数: ~{total_steps}")
    print(f"  预计用时: {total_steps * 4 / 60:.0f} - {total_steps * 6 / 60:.0f} 分钟")
    print(f"\n{'=' * 50}")
    print("开始训练...")
    print(f"{'=' * 50}")

    stats = trainer.train()

    print(f"\n{'=' * 50}")
    print(f"训练完成！")
    print(f"  总步数: {stats.global_step}")
    print(f"  最终 loss: {stats.training_loss:.4f}")
    print(f"  用时: {stats.metrics['train_runtime'] / 60:.0f} 分钟")
    print(f"{'=' * 50}")

    return model, tokenizer


def export_gguf(model, tokenizer, output_dir: str, quant_method: str = "q4_k_m"):
    """导出为 GGUF 格式。"""
    gguf_dir = os.path.join(output_dir, "gguf")

    print(f"\n[导出] 正在导出 GGUF（{quant_method}）...")
    print("  这一步需要 10-15 分钟，请耐心等待。")

    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant_method)

    gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
    if gguf_files:
        for f in gguf_files:
            size_mb = os.path.getsize(f) / 1024 / 1024
            print(f"  ✅ {os.path.basename(f)} ({size_mb:.0f} MB)")
    else:
        print("  [警告] 未找到 GGUF 文件，请检查导出过程。")

    return gguf_dir


def export_lora(model, tokenizer, output_dir: str):
    """保存 LoRA 适配器。"""
    lora_dir = os.path.join(output_dir, "lora_adapter")
    print(f"\n[导出] 保存 LoRA 适配器到 {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print("  ✅ LoRA 适配器已保存")
    return lora_dir


def main():
    parser = argparse.ArgumentParser(description="和菜头风格 LoRA 微调训练")
    parser.add_argument("--data-dir", default="training_data", help="训练数据目录")
    parser.add_argument("--output-dir", default="hecaitou_output", help="输出目录")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="基座模型")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--batch-size", type=int, default=1, help="批大小")
    parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA 秩")
    parser.add_argument("--quant", default="q4_k_m", help="GGUF 量化方法")
    parser.add_argument("--no-export", action="store_true", help="训练完不导出 GGUF")
    parser.add_argument("--export-only", action="store_true", help="只导出，不训练")
    parser.add_argument("--model-dir", default=None, help="已训练模型目录（配合 --export-only）")
    args = parser.parse_args()

    # 环境检查
    if not check_environment():
        sys.exit(1)

    if args.export_only:
        # 只导出模式
        if not args.model_dir:
            print("[错误] --export-only 需要配合 --model-dir 指定已训练模型路径")
            sys.exit(1)
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_dir,
            max_seq_length=args.max_seq_len,
            load_in_4bit=True,
        )
        export_gguf(model, tokenizer, args.output_dir, args.quant)
        return

    # 正常训练流程
    train_dataset, eval_dataset = load_data(args.data_dir)
    model, tokenizer = load_model(args.model, args.max_seq_len, args.lora_r)
    model, tokenizer = train(model, tokenizer, train_dataset, eval_dataset, args)

    # 导出
    export_lora(model, tokenizer, args.output_dir)
    if not args.no_export:
        export_gguf(model, tokenizer, args.output_dir, args.quant)

    print(f"\n{'=' * 60}")
    print("🎉 全部完成！")
    print(f"\n  GGUF 模型: {args.output_dir}/gguf/*.gguf")
    print(f"  LoRA 适配器: {args.output_dir}/lora_adapter/")
    print(f"\n下一步：")
    print(f"  1. 创建 Modelfile（见 FINETUNE_GUIDE.md Step 5）")
    print(f"  2. ollama create hecaitou-writer -f Modelfile")
    print(f"  3. ollama run hecaitou-writer '写一篇关于...'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
