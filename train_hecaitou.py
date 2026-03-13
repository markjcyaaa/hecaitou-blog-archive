#!/usr/bin/env python3
"""
和菜头风格 LoRA 微调训练脚本 v4.3
==================================
基于 Unsloth + Qwen3.5 + bf16 LoRA，使用和菜头 1060 篇文章进行风格微调。

v4.3 修复（2026-03-14）：
  - 修复 RuntimeError "You must specify a formatting_func"
    根因：Unsloth 封装的 UnslothSFTTrainer 在 _prepare_dataset 阶段会强制要求
    formatting_func，不走 TRL 原生的 prompt/completion 自动路径。
  - 修复方案：在进入 SFTTrainer 之前，用 convert_prompt_completion_to_text()
    将 prompt/completion 数据集预处理为完整 ChatML 文本（text 字段），
    然后走 dataset_text_field="text" 路径，Unsloth 完全兼容。
  - Loss 说明：全序列 loss（prompt+completion 均参与）。
    对于"学风格"任务影响极小，因为 prompt 是短指令，completion 是长正文，
    梯度主体仍来自 completion。
  - evaluate_test_set 同步修复，与训练路径一致。

v4.2 修复（2026-03-13）：
  - 彻底修复 completion-only loss 兼容性（TRL 原生方案，但 Unsloth 层拦截）

v4.1 修复（2026-03-13）：
  - 修复 RuntimeError "You must specify a formatting_func" 的根因

v4.0 更新（2026-03-13）：
  - train/val/test 三路数据：训练用 train，中间评估用 val，最终评估用 test
  - 默认超参：lr=2e-4, epochs=2, lora_r=16, lora_alpha=16, lora_dropout=0,
    weight_decay=0.01, max_seq_length=4096

使用环境：
  - 本地 NVIDIA GPU 8GB (RTX 5060 等) → Qwen3.5-2B bf16 LoRA（推荐）
  - Google Colab (免费 T4 GPU, 16GB) → Qwen3.5-4B bf16 LoRA
  - 本地 NVIDIA GPU 22GB+ (RTX 4090 等) → Qwen3.5-9B bf16 LoRA

用法：
  # 最简用法（自动检测 GPU 并选择最优配置）
  python train_hecaitou.py

  # RTX 5060 推荐配置
  python train_hecaitou.py --model Qwen/Qwen3.5-2B --max-seq-len 4096

  # 基线对照实验
  python train_hecaitou.py --data-dir training_data_baseline --output-dir output_baseline --experiment-name baseline
  python train_hecaitou.py --data-dir training_data_improved --output-dir output_improved --experiment-name improved

  # 从检查点恢复训练
  python train_hecaitou.py --resume

依赖安装（Windows Conda 方式 - 推荐）：
  conda create --name unsloth_env python==3.12 -y
  conda activate unsloth_env
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip install unsloth
"""

import argparse
import collections
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
    print("和菜头风格 LoRA 微调训练脚本 v4.3")
    print("=" * 60)

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
    
    v4.0 默认超参：
    - lora_r=16, lora_alpha=16（或用户指定）
    - bf16 LoRA（不推荐 QLoRA 4-bit）
    - max_seq_len 优先覆盖训练数据
    """
    mem = gpu_info["gpu_mem_gb"]

    if user_model:
        model = user_model
    elif mem >= 22:
        model = "Qwen/Qwen3.5-9B"
    elif mem >= 10:
        model = "Qwen/Qwen3.5-4B"
    elif mem >= 5:
        model = "Qwen/Qwen3.5-2B"
    else:
        model = "Qwen/Qwen3.5-0.8B"

    is_9b = "9B" in model or "9b" in model
    is_4b = "4B" in model or "4b" in model
    is_2b = "2B" in model or "2b" in model

    if user_lora_r:
        lora_r = user_lora_r
    else:
        lora_r = 16

    if user_seq_len:
        max_seq_len = user_seq_len
    elif is_2b and mem >= 7:
        max_seq_len = 4096
    elif is_2b:
        max_seq_len = 2048
    elif is_4b and mem >= 14:
        max_seq_len = 4096
    elif is_4b:
        max_seq_len = 2048
    elif is_9b:
        max_seq_len = 2048
    else:
        max_seq_len = 4096

    print(f"\n[自动配置] 模型: {model}")
    print(f"[自动配置] 训练模式: bf16 LoRA（Qwen3.5 官方推荐）")
    print(f"[自动配置] LoRA r: {lora_r}, alpha: {lora_r}")
    print(f"[自动配置] 最大序列长度: {max_seq_len}")
    print(f"[自动配置] 显存预估: {'充足' if mem >= 12 else '可用（已优化参数）'}")

    return model, lora_r, max_seq_len


# ============================================================
# 数据加载
# ============================================================

def detect_data_format(data_dir: str) -> str:
    """自动检测数据格式。"""
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
    except Exception:
        return "unknown"


def load_data(data_dir: str):
    """加载 train/val/test 数据，自动检测格式。"""
    from datasets import load_dataset

    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")
    eval_file = os.path.join(data_dir, "eval.jsonl")  # 向后兼容
    test_file = os.path.join(data_dir, "test.jsonl")

    if not os.path.exists(train_file):
        print(f"\n[错误] 未找到训练数据: {train_file}")
        print("  请先运行: python prepare_training_data.py")
        sys.exit(1)

    data_format = detect_data_format(data_dir)
    print(f"\n[数据] 检测到格式: {data_format}")

    if data_format == "messages":
        print("\n  ⚠️  检测到旧版 messages 格式！")
        print("  建议迁移到 v4.0 的 prompt/completion 格式：")
        print("    python prepare_training_data.py  # 重新生成数据")
        print("  当前将自动转换为纯文本格式继续训练（全序列 loss）。\n")

    train_dataset = load_dataset("json", data_files=train_file, split="train")

    val_dataset = None
    if os.path.exists(val_file):
        val_dataset = load_dataset("json", data_files=val_file, split="train")
    elif os.path.exists(eval_file):
        val_dataset = load_dataset("json", data_files=eval_file, split="train")

    test_dataset = None
    if os.path.exists(test_file):
        test_dataset = load_dataset("json", data_files=test_file, split="train")

    print(f"[数据] 训练集: {len(train_dataset)} 条")
    if val_dataset:
        print(f"[数据] 验证集: {len(val_dataset)} 条")
    if test_dataset:
        print(f"[数据] 测试集: {len(test_dataset)} 条")

    print_category_distribution(train_file, "训练集")

    return train_dataset, val_dataset, test_dataset, data_format


def print_category_distribution(jsonl_path: str, label: str):
    """从 JSONL 文件中统计并打印类别分布。"""
    CATEGORY_LABELS = {
        "A": "社会观察", "B": "技术产品评论", "C": "生死无常感悟",
        "D": "自省修行", "E": "文化阅读评论", "F": "日常生活随笔",
    }
    cat_counts = collections.Counter()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompt_content = ""
                if "prompt" in data:
                    prompt_val = data["prompt"]
                    if isinstance(prompt_val, str):
                        prompt_content = prompt_val
                    elif isinstance(prompt_val, list):
                        for msg in prompt_val:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                prompt_content = msg.get("content", "")
                                break
                elif "messages" in data:
                    for msg in data["messages"]:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            prompt_content = msg.get("content", "")
                            break

                matched = False
                for code, label_name in CATEGORY_LABELS.items():
                    if label_name in prompt_content:
                        cat_counts[code] += 1
                        matched = True
                        break
                if not matched:
                    cat_counts["?"] += 1
    except Exception:
        return

    if cat_counts:
        total = sum(cat_counts.values())
        print(f"[数据] {label} 类别分布:")
        for code in sorted(cat_counts.keys()):
            cnt = cat_counts[code]
            name = CATEGORY_LABELS.get(code, "未知")
            print(f"  {code} {name}: {cnt} ({cnt/total*100:.1f}%)")


def convert_messages_to_text(dataset, tokenizer):
    """将旧版 ChatML messages 格式转为纯文本。"""
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
        return {"text": "\n".join(parts)}

    converted = dataset.map(convert_fn, remove_columns=dataset.column_names)
    print(f"  已转换 {len(converted)} 条 messages → 纯文本")
    return converted


def convert_prompt_completion_to_text(dataset, tokenizer):
    """
    v4.3 核心修复：将 prompt/completion 格式预处理为完整 ChatML text。

    根因说明：
      Unsloth 封装的 UnslothSFTTrainer._prepare_dataset() 会强制要求
      formatting_func，不走 TRL 原生的 prompt/completion 自动路径，
      导致 RuntimeError: "You must specify a formatting_func"。

    修复方案：
      在进入 SFTTrainer 之前，把每条样本的 prompt + completion 用
      tokenizer.apply_chat_template() 拼成完整文本，存入 "text" 字段。
      然后走 dataset_text_field="text" 路径，Unsloth 完全兼容。

    数据格式约定（prepare_training_data.py v4.x 输出）：
      prompt  = "系统指令\n\n用户请求"（纯自然语言，不含特殊 token）
      completion = 文章正文（纯自然语言）

    组装策略：
      把整个 prompt 当作 user 消息，completion 当作 assistant 消息，
      用 tokenizer 的 chat_template 组装为标准 ChatML 序列。
      如果 tokenizer 没有 chat_template，退回到手动拼接。
    """
    SYSTEM = (
        "你是和菜头，运营公众号「槽边往事」。"
        "写作风格：温和的刻薄，冷幽默，短句为主，善用自嘲和比喻，第一人称，结尾留余味。"
    )

    def to_text(example):
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")

        # 尝试用 tokenizer.apply_chat_template 组装
        try:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback：手动拼接 ChatML
            text = (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{completion}<|im_end|>"
            )

        return {"text": text}

    converted = dataset.map(to_text, remove_columns=dataset.column_names)
    print(f"  已转换 {len(converted)} 条 prompt/completion → ChatML text")
    return converted


# ============================================================
# 模型加载
# ============================================================

def load_model(model_name: str, max_seq_length: int, lora_r: int, lora_alpha: int):
    """加载模型并配置 bf16 LoRA。"""
    import torch
    from unsloth import FastLanguageModel

    print(f"\n[模型] 加载 {model_name}（bf16 LoRA）...")
    print(f"  这一步需要下载模型文件，首次可能较慢...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    print(f"\n[模型] 配置 LoRA（r={lora_r}, alpha={lora_alpha}, dropout=0）...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )

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

def run_training(model, tokenizer, train_dataset, val_dataset, data_format, args, gpu_info):
    """执行训练。"""
    import torch
    from trl import SFTTrainer, SFTConfig

    use_bf16 = gpu_info["bf16"]

    print(f"\n[训练] 配置:")
    print(f"  实验名称: {args.experiment_name}")
    print(f"  模型精度: {'bf16' if use_bf16 else 'float32 (Unsloth 自动管理)'}")
    print(f"  数据格式: {data_format}")
    print(f"  轮次 (epochs): {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  等效 batch size: {args.batch_size * args.grad_accum}")
    print(f"  最大序列长度: {args.max_seq_len}")

    sft_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": 20,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "fp16": False,
        "bf16": use_bf16,
        "seed": 42,
        "report_to": "none",
        "max_seq_length": args.max_seq_len,
        # v4.3: 统一走 dataset_text_field="text" 路径，兼容 Unsloth
        "dataset_text_field": "text",
    }

    print(f"\n  [Loss 路径确认]")
    if data_format == "text":
        print(f"    ✅ 数据集格式: ChatML text（全序列 loss）")
        print(f"    ℹ️  prompt(指令)+completion(正文) 均参与 loss，正文占比 >90%，风格学习正常")
    else:
        print(f"    ✅ 数据集格式: {data_format} → 已转换为 ChatML text")
        print(f"    ℹ️  全序列 loss，Unsloth 兼容路径")
    print(f"    ❌ 未使用 formatting_func（Unsloth 不兼容）")
    print(f"    ❌ 未使用 completion_only_loss（Unsloth 封装层拦截）")

    sft_config = SFTConfig(**sft_kwargs)

    if val_dataset:
        sft_config.eval_strategy = "steps"
        sft_config.eval_steps = 100

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    total_steps = len(train_dataset) * args.epochs // (args.batch_size * args.grad_accum)
    est_min_low = total_steps * 4 / 60
    est_min_high = total_steps * 6 / 60
    print(f"\n  预计总步数: ~{total_steps}")
    print(f"  预计用时: {est_min_low:.0f} - {est_min_high:.0f} 分钟")

    print(f"\n  [过拟合判据]")
    print(f"    ❌ train_loss < 0.3 且 val_loss > 1.5 → 严重过拟合")
    print(f"    ❌ val_loss 连续 3 个 eval 不降 → 应停止训练")
    print(f"    ❌ 生成文本出现大段原文复读 → 过拟合")
    print(f"    ✅ train_loss 在 0.5-1.0，val_loss 在 0.8-1.5 → 正常范围")

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
    print(f"  实验: {args.experiment_name}")
    print(f"  总步数: {stats.global_step}")
    print(f"  最终 train loss: {stats.training_loss:.4f}")
    print(f"  实际用时: {elapsed:.0f} 分钟")
    print(f"{'=' * 50}")

    log_path = os.path.join(args.output_dir, "training_log.json")
    os.makedirs(args.output_dir, exist_ok=True)
    log_data = {
        "experiment_name": args.experiment_name,
        "model": args.model or "auto",
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "weight_decay": args.weight_decay,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size * args.grad_accum,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "data_format": data_format,
        "total_steps": stats.global_step,
        "final_train_loss": stats.training_loss,
        "training_minutes": round(elapsed, 1),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"  训练日志: {log_path}")

    return model, tokenizer


# ============================================================
# 测试集评估
# ============================================================

def evaluate_test_set(model, tokenizer, test_dataset, data_format, args):
    """在测试集上评估，输出整体 loss。"""
    import torch
    from trl import SFTTrainer, SFTConfig

    print(f"\n{'=' * 50}")
    print(f"[测试集评估]")

    sft_kwargs = {
        "output_dir": os.path.join(args.output_dir, "test_eval"),
        "per_device_eval_batch_size": 1,
        "fp16": False,
        "bf16": torch.cuda.is_bf16_supported(),
        "report_to": "none",
        "max_seq_length": args.max_seq_len,
        # v4.3: 统一走 dataset_text_field="text"
        "dataset_text_field": "text",
    }

    sft_config = SFTConfig(**sft_kwargs)
    sft_config.eval_strategy = "no"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=test_dataset,
        args=sft_config,
    )

    metrics = trainer.evaluate(eval_dataset=test_dataset)
    test_loss = metrics.get("eval_loss", "N/A")
    print(f"  整体 test loss: {test_loss}")

    result_path = os.path.join(args.output_dir, "test_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": args.experiment_name, "test_loss": test_loss, **metrics},
                  f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {result_path}")
    print(f"{'=' * 50}")


# ============================================================
# 生成示例评估
# ============================================================

def generate_evaluation_samples(model, tokenizer, args):
    """生成几个评估示例，用于人工判断风格质量。"""
    from unsloth import FastLanguageModel

    print(f"\n[生成评估] 生成 6 个类别各 1 个示例...")

    EVAL_PROMPTS = {
        "A": "[社会观察] 写一篇关于「当代年轻人的疲惫」的文章",
        "B": "[技术产品评论] 写一篇关于「为什么我不买最新款手机」的文章",
        "C": "[生死无常感悟] 写一篇关于「远行」的文章",
        "D": "[自省修行] 写一篇关于「与自己和解」的文章",
        "E": "[文化阅读评论] 写一篇关于「重读一本旧书」的文章",
        "F": "[日常生活随笔] 写一篇关于「今天什么都不想干」的文章",
    }

    SYSTEM = "你是和菜头，运营公众号「槽边往事」。写作风格：温和的刻薄，冷幽默，短句为主，善用自嘲和比喻，第一人称，结尾留余味。"

    FastLanguageModel.for_inference(model)

    results = {}
    for cat, user_prompt in EVAL_PROMPTS.items():
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        import torch
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        results[cat] = {"prompt": user_prompt, "response": response[:500]}
        print(f"\n  [{cat}] {user_prompt}")
        print(f"  → {response[:200]}...")

    eval_path = os.path.join(args.output_dir, "eval_samples.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  评估样本已保存: {eval_path}")

    FastLanguageModel.for_training(model)


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
        return None

    gguf_files = glob.glob(os.path.join(gguf_dir, "*.gguf"))
    if gguf_files:
        for f in gguf_files:
            size_mb = os.path.getsize(f) / 1024 / 1024
            print(f"  {os.path.basename(f)} ({size_mb:.0f} MB)")
        return gguf_dir
    else:
        print("  [警告] 未找到 GGUF 文件。")
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
        description="和菜头风格 LoRA 微调训练 v4.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 最简用法（自动检测 GPU 并选择最优配置）
  python train_hecaitou.py

  # RTX 5060 (8GB) 推荐配置
  python train_hecaitou.py --model Qwen/Qwen3.5-2B --max-seq-len 4096

  # 基线对照实验
  python train_hecaitou.py --data-dir training_data_baseline \\
      --output-dir output_baseline --experiment-name baseline
  python train_hecaitou.py --data-dir training_data_improved \\
      --output-dir output_improved --experiment-name improved

  # 从中断处恢复
  python train_hecaitou.py --resume

  # 只导出 GGUF
  python train_hecaitou.py --export-only --model-dir ./hecaitou_output/lora_adapter
        """,
    )
    parser.add_argument("--data-dir", default="training_data",
                        help="训练数据目录（默认 training_data/）")
    parser.add_argument("--output-dir", default="hecaitou_output",
                        help="输出目录（默认 hecaitou_output/）")
    parser.add_argument("--model", default=None,
                        help="基座模型（默认自动选择）")
    parser.add_argument("--epochs", type=int, default=2,
                        help="训练轮次（默认 2）")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率（默认 2e-4）")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="批大小（默认 1）")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="梯度累积步数（默认 4，等效 batch=4）")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="最大序列长度（默认自动）")
    parser.add_argument("--lora-r", type=int, default=None,
                        help="LoRA 秩（默认 16）")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha（默认与 lora-r 相同）")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="权重衰减（默认 0.01）")
    parser.add_argument("--quant", default="q4_k_m",
                        help="GGUF 量化方法（默认 q4_k_m）")
    parser.add_argument("--no-export", action="store_true",
                        help="训练完不导出 GGUF（只保存 LoRA）")
    parser.add_argument("--export-only", action="store_true",
                        help="只导出 GGUF，不训练")
    parser.add_argument("--model-dir", default=None,
                        help="已训练模型目录（配合 --export-only）")
    parser.add_argument("--resume", action="store_true",
                        help="从最近的检查点恢复训练")
    parser.add_argument("--cache-dir", default=None,
                        help="模型缓存目录")
    parser.add_argument("--experiment-name", default="default",
                        help="实验名称（用于区分 baseline/improved，默认 default）")
    parser.add_argument("--skip-test-eval", action="store_true",
                        help="跳过测试集评估")
    parser.add_argument("--skip-generate-eval", action="store_true",
                        help="跳过生成评估样本")
    args = parser.parse_args()

    # ---- 设置缓存目录 ----
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
    if args.lora_r is None:
        args.lora_r = lora_r
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r

    # ---- 加载数据 ----
    train_dataset, val_dataset, test_dataset, data_format = load_data(args.data_dir)

    # ---- 加载模型 ----
    model, tokenizer = load_model(model_name, max_seq_len, args.lora_r, args.lora_alpha)

    # ---- 数据预处理（统一转为 ChatML text，兼容 Unsloth）----
    if data_format == "prompt_completion":
        print("\n[数据] prompt/completion 格式 → 转换为 ChatML text（v4.3 修复）")
        train_dataset = convert_prompt_completion_to_text(train_dataset, tokenizer)
        if val_dataset:
            val_dataset = convert_prompt_completion_to_text(val_dataset, tokenizer)
        if test_dataset:
            test_dataset = convert_prompt_completion_to_text(test_dataset, tokenizer)
        data_format = "text"
    elif data_format == "messages":
        print("\n[数据] 检测到旧版 messages 格式，转换为纯文本...")
        train_dataset = convert_messages_to_text(train_dataset, tokenizer)
        if val_dataset:
            val_dataset = convert_messages_to_text(val_dataset, tokenizer)
        if test_dataset:
            test_dataset = convert_messages_to_text(test_dataset, tokenizer)
        data_format = "text"
    elif data_format == "text":
        print("\n[数据] text 格式，直接使用")

    # ---- 训练 ----
    model, tokenizer = run_training(
        model, tokenizer, train_dataset, val_dataset, data_format, args, gpu_info
    )

    # ---- 测试集评估 ----
    if test_dataset and not args.skip_test_eval:
        evaluate_test_set(model, tokenizer, test_dataset, data_format, args)

    # ---- 生成评估样本 ----
    if not args.skip_generate_eval:
        try:
            generate_evaluation_samples(model, tokenizer, args)
        except Exception as e:
            print(f"  [跳过] 生成评估样本失败: {e}")

    # ---- 导出 ----
    export_lora(model, tokenizer, args.output_dir)

    if not args.no_export:
        gguf_dir = export_gguf(model, tokenizer, args.output_dir, args.quant)
    else:
        gguf_dir = None

    # ---- 完成 ----
    print(f"\n{'=' * 60}")
    print("全部完成！")
    print(f"\n  实验: {args.experiment_name}")
    print(f"  LoRA 适配器: {args.output_dir}/lora_adapter/")
    if gguf_dir:
        print(f"  GGUF 模型: {args.output_dir}/gguf/*.gguf")

    print(f"\n下一步：")
    if gguf_dir:
        print(f"  ollama create hecaitou-writer -f Modelfile")
        print(f"  ollama run hecaitou-writer '写一篇关于...'")
    else:
        print(f"  1. 合并 LoRA:")
        print(f"     见 FINETUNE_GUIDE.md Step 4A 方案二")

    if args.experiment_name == "default":
        print(f"\n对照实验（可选）：")
        print(f"  python prepare_training_data.py --baseline --output training_data_baseline")
        print(f"  python train_hecaitou.py --data-dir training_data_baseline "
              f"--output-dir output_baseline --experiment-name baseline")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
