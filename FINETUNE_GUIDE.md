# 和菜头风格 LoRA 微调：从零到部署完整指南

> **适用对象**：微调纯新手，无 GPU 也能完成
> **总用时**：3-5 小时（含等待训练时间）
> **费用**：0 元（使用 Google Colab 免费 GPU）
> **最终产物**：一个 GGUF 模型文件，导入 Ollama 直接用

---

## 全流程概览

```
你现在在这里
      |
      v
[Step 0] 理解原理（5 分钟阅读）
      |
[Step 1] 本地生成训练数据（2 分钟）
      |
[Step 2] 上传数据到 Google Drive（5 分钟）
      |
[Step 3] 打开 Colab，粘贴训练代码，点运行（10 分钟操作，2-3 小时训练）
      |
[Step 4] 从 Colab 导出 GGUF 模型文件（10 分钟）
      |
[Step 5] 下载到本地，导入 Ollama（10 分钟）
      |
[Step 6] 测试效果，替换 hecaitou_writer.py 的模型
      |
      v
   完成 🎉
```

---

## Step 0：5 分钟理解原理（可跳过，但建议读）

### 什么是 LoRA 微调？

你现在的方案是：用 System Prompt（1000 字规则）告诉通用模型"请像和菜头那样写"。这就像给一个外国人一张中餐菜谱，让他照着做——能做出来，但味道差很远。

**LoRA 微调**是让模型直接读 1062 篇和菜头的文章，从参数层面学会他的写作节奏。这就像让那个外国人在中餐馆当了三个月学徒——他不需要看菜谱了，因为手感已经在肌肉里了。

### 为什么选 Qwen3.5-9B 而不是你现在的 35B？

| | Qwen3.5-35B (你现在用的) | Qwen3.5-9B (微调目标) |
|---|---|---|
| 架构 | MoE (混合专家) | Dense (稠密) |
| 实际激活参数 | ~3.5B | 9B |
| Unsloth 微调支持 | ❌ 不支持 MoE | ✅ 完整支持 |
| 推理速度 | 较慢（加载 35B 权重） | 更快 |
| 本地运行内存 | ~20GB | ~9GB (Q4 量化后) |
| 微调后文字质量 | 无法微调 | **远超 prompt engineering** |

简单说：9B dense 微调后的效果 > 35B MoE + 精心 prompt。

### 为什么免费？

Google Colab 免费提供 T4 GPU（16GB 显存），足够训练 9B 模型的 QLoRA（4-bit 量化 LoRA）。每次 session 最长约 12 小时，我们的训练大约需要 2-3 小时，完全在免费额度内。

---

## Step 1：本地生成训练数据（2 分钟）

### 1.1 打开 PowerShell，进入项目目录

```powershell
cd C:\Users\hzjin\hecaitou-blog-archive
```

### 1.2 运行数据准备脚本

```powershell
python prepare_training_data.py
```

你会看到类似这样的输出：

```
[Step 1] 加载分类文章...
  A_社会观察.md: 195 篇
  B_技术产品.md: 219 篇
  ...
  共加载 1067 篇文章

[Step 3] 生成训练样本...
  总样本数: 1062
  预估总 tokens: 2,906,847

[Step 4] 划分数据集...
  训练集: 956 条
  验证集: 106 条

[Step 5] 文件已写入:
  training_data/train.jsonl  (5.6 MB)
  training_data/eval.jsonl  (673 KB)

✅ 数据准备完成！
```

### 1.3 可选：预览生成的数据

```powershell
python prepare_training_data.py --preview 3
```

### 1.4 检查输出文件

```powershell
dir training_data\
```

应该看到三个文件：
- `train.jsonl` — 训练集（约 5.6 MB）
- `eval.jsonl` — 验证集（约 670 KB）
- `data_report.txt` — 统计报告

---

## Step 2：上传数据到 Google Drive（5 分钟）

### 2.1 打开 Google Drive

浏览器访问 https://drive.google.com

### 2.2 创建文件夹

在 Google Drive 根目录创建一个文件夹，命名为：`hecaitou_finetune`

### 2.3 上传训练数据

把本地 `training_data/` 目录下的两个文件拖进去：
- `train.jsonl`
- `eval.jsonl`

上传完成后你的 Google Drive 结构应该是：

```
我的云端硬盘/
  └── hecaitou_finetune/
        ├── train.jsonl
        └── eval.jsonl
```

---

## Step 3：在 Colab 中训练（核心步骤）

### 3.1 打开 Google Colab

浏览器访问 https://colab.research.google.com

### 3.2 新建笔记本

点击左上角 **"文件" → "新建笔记本"**

### 3.3 切换到 GPU 运行时

这一步非常重要：

1. 点击菜单栏 **"运行时" (Runtime) → "更改运行时类型" (Change runtime type)**
2. 在弹出的对话框中：
   - **硬件加速器** 选择 **T4 GPU**
   - 点击 **保存**

### 3.4 粘贴以下代码到笔记本

整个训练流程分为 5 个代码块。你需要创建 5 个代码单元格（点击 **"+ 代码"**），依次粘贴并运行。

---

#### 代码块 1：安装依赖（运行约 3-5 分钟）

```python
# ===== 代码块 1/5：安装 Unsloth =====
# 这一步安装所有需要的库，需要等几分钟

!pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo 2>&1 | tail -5
!pip install --upgrade trl datasets 2>&1 | tail -3

# 验证安装
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
```

**预期输出**：显示 `CUDA 可用: True`，GPU 为 `Tesla T4`，显存 `15.x GB`。

如果显示 `CUDA 可用: False`，说明运行时没有切换到 GPU，请回到 3.3 重新设置。

---

#### 代码块 2：挂载 Google Drive + 加载数据（运行约 1 分钟）

```python
# ===== 代码块 2/5：加载训练数据 =====
from google.colab import drive
from datasets import load_dataset

# 挂载 Google Drive（会弹出授权窗口，点击允许）
drive.mount('/content/drive')

# 加载数据
data_dir = "/content/drive/MyDrive/hecaitou_finetune"
train_dataset = load_dataset("json", data_files=f"{data_dir}/train.jsonl", split="train")
eval_dataset = load_dataset("json", data_files=f"{data_dir}/eval.jsonl", split="train")

print(f"训练集: {len(train_dataset)} 条")
print(f"验证集: {len(eval_dataset)} 条")
print(f"\n样本示例:")
print(train_dataset[0]["messages"][1])  # 打印第一条的用户指令
```

**注意**：运行这一块时，Colab 会弹出 Google 账号授权窗口，点击"允许"即可。

---

#### 代码块 3：加载模型 + 配置 LoRA（运行约 5 分钟）

```python
# ===== 代码块 3/5：加载 Qwen3.5-9B + 配置 LoRA =====
from unsloth import FastLanguageModel

# 加载 Qwen3.5-9B（4-bit 量化，省显存）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3.5-9B",
    max_seq_length = 2048,
    load_in_4bit = True,        # QLoRA：4-bit 量化加载
    full_finetuning = False,    # 不做全量微调
)

# 给模型挂上 LoRA 适配器
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                     # LoRA 秩（越大学习能力越强，显存占用越多）
    target_modules = [          # 要微调的层
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32,            # 缩放因子，通常和 r 相同
    lora_dropout = 0,           # 不用 dropout
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth 独家优化，大幅降低显存
    random_state = 42,
    max_seq_length = 2048,
)

# 显示可训练参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n模型总参数: {total_params / 1e9:.2f}B")
print(f"可训练参数: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.2f}%)")
print(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

**预期输出**：可训练参数约 100-200M（占总参数 1-2%），显存占用约 6-8 GB。

---

#### 代码块 4：开始训练（运行约 2-3 小时）

```python
# ===== 代码块 4/5：开始训练 =====
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = SFTConfig(
        # ---- 基本参数 ----
        output_dir = "/content/hecaitou_output",    # 训练输出目录
        max_seq_length = 2048,                      # 最大序列长度

        # ---- 训练量 ----
        num_train_epochs = 3,                       # 训练 3 轮（epoch）
        per_device_train_batch_size = 1,            # 每次处理 1 条（T4 显存限制）
        gradient_accumulation_steps = 4,            # 累积 4 步 = 等效 batch_size 4

        # ---- 学习率 ----
        learning_rate = 2e-4,                       # 学习率
        warmup_steps = 20,                          # 前 20 步热身
        optim = "adamw_8bit",                       # 8-bit 优化器，省显存
        weight_decay = 0.01,

        # ---- 日志 ----
        logging_steps = 10,                         # 每 10 步打印一次 loss
        eval_strategy = "steps",                    # 按步数评估
        eval_steps = 100,                           # 每 100 步评估一次
        save_strategy = "steps",                    # 按步数保存
        save_steps = 200,                           # 每 200 步保存检查点

        # ---- 其他 ----
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        seed = 42,
        report_to = "none",                         # 不上报到 wandb
    ),
)

# 打印训练计划
total_steps = len(train_dataset) * 3 // (1 * 4)  # samples × epochs / (batch × accum)
print(f"训练计划:")
print(f"  样本数: {len(train_dataset)}")
print(f"  轮次 (epochs): 3")
print(f"  等效 batch size: 4")
print(f"  预计总步数: ~{total_steps}")
print(f"  预计用时: {total_steps * 4 / 60:.0f} - {total_steps * 6 / 60:.0f} 分钟")
print(f"\n开始训练...")
print("=" * 50)

# 开始！
trainer_stats = trainer.train()

# 训练完成
print("=" * 50)
print(f"训练完成！")
print(f"  总步数: {trainer_stats.global_step}")
print(f"  最终 loss: {trainer_stats.training_loss:.4f}")
print(f"  总用时: {trainer_stats.metrics['train_runtime'] / 60:.0f} 分钟")
```

**这一步需要耐心等待**。你会看到 loss 数字不断打印出来：

```
Step 10: loss = 2.1234
Step 20: loss = 1.8765
Step 30: loss = 1.5432
...
```

**loss 持续下降 = 正在学习**。正常的最终 loss 大约在 0.8-1.2 之间。

> **中间可以干别的**：训练过程中不需要盯着屏幕。但注意 **不要关闭浏览器标签页**，否则 Colab 会断开连接。可以把浏览器窗口最小化，该干嘛干嘛。

---

#### 代码块 5：导出模型为 GGUF（运行约 10-15 分钟）

```python
# ===== 代码块 5/5：导出 GGUF 模型 =====
# 导出为 GGUF 格式（Ollama / llama.cpp 直接可用）
print("正在导出 GGUF（Q4_K_M 量化）...")
print("这一步需要 10-15 分钟，请耐心等待。")

model.save_pretrained_gguf(
    "/content/hecaitou_gguf",
    tokenizer,
    quantization_method = "q4_k_m"      # Q4 量化，9GB 内存可跑
)

print("GGUF 导出完成！")

# 复制到 Google Drive（防止 Colab 断开后丢失）
import shutil, glob

gguf_files = glob.glob("/content/hecaitou_gguf/*.gguf")
drive_output = "/content/drive/MyDrive/hecaitou_finetune/output"
!mkdir -p "{drive_output}"

for f in gguf_files:
    print(f"复制到 Google Drive: {f}")
    shutil.copy2(f, drive_output)

# 同时保存 LoRA 适配器（体积小，方便以后继续训练）
model.save_pretrained("/content/drive/MyDrive/hecaitou_finetune/output/lora_adapter")
tokenizer.save_pretrained("/content/drive/MyDrive/hecaitou_finetune/output/lora_adapter")

print("\n" + "=" * 50)
print("✅ 全部完成！文件已保存到 Google Drive:")
print(f"   GGUF 模型: hecaitou_finetune/output/*.gguf")
print(f"   LoRA 适配器: hecaitou_finetune/output/lora_adapter/")
print("=" * 50)

# 列出输出文件
!ls -lh /content/drive/MyDrive/hecaitou_finetune/output/*.gguf
```

---

### 3.5 完成训练后

如果一切顺利，你的 Google Drive 中会多出：

```
hecaitou_finetune/
  ├── train.jsonl              (你上传的)
  ├── eval.jsonl               (你上传的)
  └── output/
        ├── hecaitou_gguf-Q4_K_M.gguf    ← 这就是你的模型（约 5-6 GB）
        └── lora_adapter/                 ← LoRA 权重备份
```

---

## Step 4：下载模型到本地（10 分钟）

### 4.1 从 Google Drive 下载

1. 打开 Google Drive
2. 进入 `hecaitou_finetune/output/`
3. 右键点击 `.gguf` 文件 → **下载**
4. 保存到本地，比如 `C:\Users\hzjin\models\`

> 文件约 5-6 GB，下载时间取决于你的网速。

---

## Step 5：导入 Ollama 并运行（10 分钟）

### 5.1 创建 Modelfile

在下载的模型同目录下，创建一个文本文件 `Modelfile`（无扩展名），内容如下：

```
FROM ./hecaitou_gguf-Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER num_ctx 16384

SYSTEM "你是和菜头，运营公众号「槽边往事」。"
```

> **注意**：`FROM` 后面的文件名要和你实际下载的 `.gguf` 文件名一致。如果文件名不同，请相应修改。

### 5.2 导入 Ollama

```powershell
cd C:\Users\hzjin\models\
ollama create hecaitou-writer -f Modelfile
```

等待导入完成，你会看到 `success` 字样。

### 5.3 快速测试

```powershell
ollama run hecaitou-writer "写一篇关于'最近大家都在用AI写东西'的文章"
```

如果模型输出了一篇带有和菜头风格的文章（冷幽默、短句、自嘲、第一人称"我"），恭喜你，微调成功了！

### 5.4 切换 hecaitou_writer.py 的模型

编辑 `hecaitou_writer.py`，找到配置部分（约第 48 行），修改：

```python
_config = {
    ...
    "writer_model": "hecaitou-writer",      # 改成你的微调模型名
    "critic_model": "hecaitou-writer",      # 同上
    ...
}
```

然后运行测试：

```powershell
python hecaitou_writer.py --topic "最近大家都在用AI写东西" --quick
```

---

## Step 6：效果不好怎么办？

### 情况 1：输出全是乱码或重复

**原因**：训练轮次太多，过拟合了。

**解决**：回到 Colab，把代码块 4 中的 `num_train_epochs` 从 3 改为 1 或 2，重新训练。

### 情况 2：风格不够像

**原因**：训练不足。

**解决**：把 `num_train_epochs` 加到 4-5，或者把 `r`（LoRA 秩）从 32 加到 64。

### 情况 3：Colab 训练中途断开

**原因**：免费 Colab 有使用限制，长时间不活跃会断开。

**解决**：
- 代码块 4 中已经设置了每 200 步保存检查点
- 重新连接后，在代码块 4 中加一行：`trainer.train(resume_from_checkpoint=True)`
- 或者用 Colab 的"保持连接"技巧：按 F12 打开浏览器控制台，粘贴：
  ```javascript
  setInterval(() => document.querySelector("colab-connect-button")?.click(), 60000)
  ```

### 情况 4：想进一步提升效果

- **增大 LoRA 秩**：`r = 64`（显存够的话）
- **增大上下文**：`max_seq_length = 4096`（能覆盖更长文章）
- **数据增强**：运行 `python prepare_training_data.py --max-length 6000` 保留更多长文

---

## 常见问题

### Q: Colab 免费版够用吗？

够。T4 GPU 16GB 显存，训练 9B QLoRA 绰绰有余。免费额度足够完成 3 轮训练。

### Q: 训练完还需要 HECAITOU_STYLE_SKILLS.md 吗？

微调后，模型从参数层面学会了风格，Skill 文件的作用大幅降低。但建议保留去 AI 化规则（Step 5 自检），作为质量兜底。

### Q: 训练数据是公开的吗？我需要担心版权吗？

训练数据来自和菜头的公开博客文章。LoRA 适配器仅供个人学习使用，不要公开分享或商用。

### Q: 以后和菜头写了新文章，怎么更新？

1. 用爬虫增量爬取：`python crawler.py --month 2026-04`
2. 运行分类：`python classify_articles.py`
3. 重新生成训练数据：`python prepare_training_data.py`
4. 在 Colab 中重新训练（可以用之前的 LoRA 作为起点继续训练）

### Q: 我有 NVIDIA 显卡（如 RTX 4090），可以本地训练吗？

完全可以，甚至更方便：

```powershell
pip install unsloth unsloth_zoo trl datasets
python train_hecaitou.py
```

本地训练不需要挂载 Google Drive 那些步骤，训练脚本 `train_hecaitou.py` 在仓库中已提供。

---

## 文件清单

| 文件 | 用途 | 在哪里运行 |
|------|------|-----------|
| `prepare_training_data.py` | 生成训练数据 | 你的 Windows 本地 |
| `training_data/train.jsonl` | 训练集 | 上传到 Google Drive |
| `training_data/eval.jsonl` | 验证集 | 上传到 Google Drive |
| `train_hecaitou.py` | 本地训练脚本（有 GPU 时用） | 本地 / Colab |
| `FINETUNE_GUIDE.md` | 本文档 | - |

---

*版本 v1.0 | 2026-03-10 | 基于 Unsloth + Qwen3.5-9B + QLoRA*
