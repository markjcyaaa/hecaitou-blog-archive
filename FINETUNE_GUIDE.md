# 和菜头风格 LoRA 微调：从零到部署完整指南 v3.2

> **更新 2026-03-13 v3.2**：修复 completion-only loss 兼容性
> - 使用 TRL 原生 `SFTConfig(completion_only_loss=True)`
> - 数据格式改为纯文本 prompt/completion（不含 ChatML 特殊 token）
> - 删除 `DataCollatorForCompletionOnlyLM` 和 `formatting_func` 依赖
> - 文章级分层切分（train/val/test = 80%/10%/10%）
> - 3 种混合 Prompt 模板（含类别信号）
> - 去重、少数类别过采样、数据泄漏检查
> - 基线 vs 改进实验支持
>
> **适用对象**：微调纯新手
> **费用**：0 元（本地 GPU 或 Google Colab 免费 GPU）
> **最终产物**：一个 GGUF 模型文件，导入 Ollama 直接用

---

## 全流程概览

```
你现在在这里
      |
      v
[Step 0] 理解原理（5 分钟阅读，可跳过）
      |
[Step 1] 本地生成训练数据（2 分钟）
      |
      +--[路线 A] Windows 本地训练（推荐！有 NVIDIA 显卡即可）
      |     |
      |     [Step 2A] 安装 Conda + Unsloth（15 分钟，仅首次）
      |     |
      |     [Step 3A] 一条命令开始训练（1-3 小时等待）
      |     |
      |     [Step 4A] 导出 + 导入 Ollama（10 分钟）
      |
      +--[路线 B] Google Colab 云端训练（备选，无 GPU 时用）
      |     |
      |     [Step 2B-4B] 见文末附录
      |
      v
[Step 5] 测试效果，替换 hecaitou_writer.py 的模型
      |
      v
[Step 6] 可选：对照实验（baseline vs improved）
      |
      v
   完成！
```

---

## Step 0：5 分钟理解原理（可跳过）

### 什么是 LoRA 微调？

你现在的方案是：用 System Prompt（1000 字规则）告诉通用模型"请像和菜头那样写"。这就像给一个外国人一张中餐菜谱，让他照着做——能做出来，但味道差很远。

**LoRA 微调**是让模型直接读 1060 篇和菜头的文章，从参数层面学会他的写作节奏。这就像让那个外国人在中餐馆当了三个月学徒——他不需要看菜谱了，因为手感已经在肌肉里了。

### v3.0 升级要点

| 特性 | v2.0 | v3.0 |
|------|------|------|
| 数据切分 | train/val 随机 90/10 | train/val/test 分层 80/10/10 |
| 切分方式 | 随机打乱 | 文章级 + 时间优先 + 类别分层 |
| Prompt 模板 | 单一模板 | 3 种混合（A:B:C = 50:30:20） |
| 类别信号 | 无 | 6 类标签融入 Prompt |
| Loss 计算 | 全序列 | Completion-only（`SFTConfig(completion_only_loss=True)`）|
| 去重 | 无 | 标题 + 正文哈希去重 |
| 数据泄漏检查 | 无 | 自动检查 train/val/test 无重叠 |
| 默认 epochs | 3 | 2（降低过拟合风险） |
| 默认 LoRA r | 32 | 16（更保守，更稳定） |

### GPU 显存 vs 模型选择

| GPU | 显存 | 推荐模型 | max_seq_len | 训练时间 |
|-----|------|----------|-------------|---------|
| RTX 5060 / 4060 | 8 GB | Qwen3.5-2B bf16 LoRA | 4096 | 1-2 小时 |
| RTX 3060 12G / 4070 | 12 GB | Qwen3.5-4B bf16 LoRA | 2048-4096 | 1-3 小时 |
| RTX 4090 / A6000 | 24 GB | Qwen3.5-9B bf16 LoRA | 2048 | 2-3 小时 |
| Tesla T4 (Colab) | 16 GB | Qwen3.5-4B bf16 LoRA | 2048 | 1-3 小时 |
| 无 NVIDIA GPU | - | 用 Colab 路线 B | - | 见附录 |

> 2B 模型 + 完整长上下文（4096） >> 4B 模型 + 截断短文本（1024）。v3.0 的核心原则是**完整文本比大模型更重要**。

---

## Step 1：本地生成训练数据（2 分钟）

### 1.1 打开 Anaconda PowerShell Prompt，进入项目目录

```powershell
cd C:\Users\hzjin\hecaitou-blog-archive
```

### 1.2 拉取最新代码

```powershell
git pull origin main
```

### 1.3 运行数据准备脚本 v4.0

```powershell
python prepare_training_data.py
```

你会看到：

```
[Step 1] 加载分类文章...
  共加载 1067 篇文章

[Step 2] 去重检查...
  去重后: 1065 篇（去除标题重复 2，正文重复 0）

[Step 3] 过滤...
  保留 1060 篇

[Step 4] 分层切分（文章级，时间优先）...
  训练集: 848 篇
  验证集: 106 篇
  测试集: 106 篇
  train: A:156 | B:174 | C:36 | D:66 | E:210 | F:206
  val:   A:20  | B:22  | C:4  | D:8  | E:26  | F:26
  test:  A:19  | B:21  | C:5  | D:8  | E:27  | F:26

[Step 7] 数据泄漏检查...
  [通过] 无数据泄漏

[Step 8] 文件已写入:
  training_data/train.jsonl  (5.3 MB)
  training_data/val.jsonl    (670 KB)
  training_data/test.jsonl   (666 KB)
```

### 1.4 可选：对少数类别过采样

C 类（生死无常）只有 36 篇训练数据，如果你觉得太少：

```powershell
python prepare_training_data.py --oversample
```

这会将 C/D 类在训练集中 2x 过采样（验证/测试集保持真实分布）。

### 1.5 检查输出文件

```powershell
dir training_data\
```

应该看到：
- `train.jsonl` — 训练集（80%，约 5.3 MB）
- `val.jsonl` — 验证集（10%）
- `test.jsonl` — 测试集（10%）
- `eval.jsonl` — = val.jsonl（向后兼容）
- `data_report.txt` — 详细统计报告

---

## 路线 A：Windows 本地训练（推荐）

> 适用于有 NVIDIA 显卡的 Windows 电脑。你的 RTX 5060 (8GB) 完全够用。

### Step 2A：安装环境（仅首次，约 15 分钟）

#### 2A.1 安装 Miniconda

如果没有装过 Conda，打开 PowerShell 运行：

```powershell
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile ".\miniconda.exe"
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait
del .\miniconda.exe
```

安装完成后，关闭 PowerShell，打开 **"Anaconda Powershell Prompt"**（在开始菜单搜索）。

> 后续所有命令都在 **Anaconda Powershell Prompt** 中执行，不是普通 PowerShell。

#### 2A.2 创建隔离环境

```powershell
conda create --name unsloth_env python==3.12 -y
conda activate unsloth_env
```

#### 2A.3 安装 PyTorch（匹配你的 CUDA 版本）

```powershell
nvidia-smi  # 看右上角 CUDA Version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

**如果显示 `CUDA: False`，停下来排查。**

#### 2A.4 安装 Unsloth

```powershell
pip install unsloth
```

> **重要**：不要再单独运行 `pip install --upgrade trl datasets`！让 unsloth 管理依赖版本。

#### 2A.5 验证安装

```powershell
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

### Step 3A：开始训练（一条命令）

```powershell
cd C:\Users\hzjin\hecaitou-blog-archive
conda activate unsloth_env
python train_hecaitou.py
```

脚本会自动：
1. 检测你的 GPU（RTX 5060, 8GB）
2. 自动选择 Qwen3.5-2B（适合 8GB，bf16 LoRA）
3. 自动配置 LoRA r=16, alpha=16, max_seq_len=4096
4. 下载模型（首次约 5GB，需 10-20 分钟）
5. 启用 completion-only loss（仅学文章正文）
6. 开始训练（约 1-2 小时）
7. 在测试集上自动评估
8. 生成 6 个类别各 1 个示例

你会看到：

```
[自动配置] 模型: Qwen/Qwen3.5-2B
[自动配置] LoRA r: 16, alpha: 16
[自动配置] 最大序列长度: 4096

[数据] 训练集: 848 条
[数据] 验证集: 106 条
[数据] 测试集: 106 条

[训练] 配置:
  ✅ Loss 模式: completion-only

开始训练...
Step 10: loss = 2.1234
Step 20: loss = 1.8765
...

训练完成！
  最终 train loss: 0.85

[测试集评估]
  整体 test loss: 1.12

[生成评估] 生成 6 个类别各 1 个示例...
  [A] [社会观察] 写一篇关于「当代年轻人的疲惫」的文章
  → 年轻人最大的问题不是累，是不知道自己为什么累...
```

#### 默认超参数（v3.0 推荐值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 2 | 2 轮即可，3 轮容易过拟合 |
| lr | 2e-4 | 标准 LoRA 学习率 |
| lora_r | 16 | 保守但稳定 |
| lora_alpha | 16 | = lora_r |
| lora_dropout | 0 | Unsloth 推荐 |
| weight_decay | 0.01 | 轻微正则 |
| max_seq_len | 4096 | 覆盖 95%+ 文章 |
| batch (effective) | 4 | batch_size=1 × grad_accum=4 |

#### 自定义参数（可选）

```powershell
# 增加 LoRA 秩（更强学习能力，更慢）
python train_hecaitou.py --lora-r 32 --lora-alpha 32

# 减少训练轮次（更快但效果稍差）
python train_hecaitou.py --epochs 1

# 使用过采样数据
python prepare_training_data.py --oversample
python train_hecaitou.py
```

### Step 4A：导出并导入 Ollama

训练完成后，脚本会自动导出 GGUF 文件到 `hecaitou_output/gguf/`。

> **注意**：Windows 上 GGUF 导出可能会失败。如果看到导出失败提示，使用 LoRA 适配器方案。

#### 方案一：GGUF 导出成功

```powershell
dir hecaitou_output\gguf\
```

创建 `Modelfile`：

```text
FROM ./hecaitou_gguf-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER num_ctx 16384
SYSTEM "你是和菜头，运营公众号「槽边往事」。"
```

导入 Ollama：

```powershell
cd hecaitou_output\gguf
ollama create hecaitou-writer -f Modelfile
```

#### 方案二：GGUF 导出失败（用 Colab 转换）

LoRA 适配器在 `hecaitou_output/lora_adapter/`。上传到 Colab 转换：

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/lora_adapter",
    max_seq_length=4096,
    load_in_4bit=True,
)
model.save_pretrained_gguf("/content/hecaitou_gguf", tokenizer, quantization_method="q4_k_m")
from google.colab import files
import glob
for f in glob.glob("/content/hecaitou_gguf/*.gguf"):
    files.download(f)
```

### Step 5：测试效果

```powershell
ollama run hecaitou-writer "写一篇关于'最近大家都在用AI写东西'的文章"
```

推理时使用与训练一致的 Prompt 格式（含类别标签）效果最佳：

```
[社会观察] 写一篇关于「最近大家都在用AI写东西」的文章
```

或完整版：

```
请以和菜头的风格，写一篇社会观察类文章。
类别：社会观察
标题：最近大家都在用AI写东西
概要：AI 写作工具越来越多，但这些工具真的能代替人类写作吗？
核心意向：技术浪潮下的人文思考
```

---

## Step 6：对照实验（可选）

### 基线 vs 改进

```powershell
# 1) 生成基线数据（无 category 信号）
python prepare_training_data.py --baseline --output training_data_baseline

# 2) 生成改进数据（含 category 信号）
python prepare_training_data.py --output training_data_improved

# 3) 分别训练
python train_hecaitou.py --data-dir training_data_baseline --output-dir output_baseline --experiment-name baseline
python train_hecaitou.py --data-dir training_data_improved --output-dir output_improved --experiment-name improved

# 4) 比较结果
type output_baseline\test_results.json
type output_improved\test_results.json
type output_baseline\eval_samples.json
type output_improved\eval_samples.json
```

比较要点：
- test_loss：改进版应更低
- eval_samples.json：对比 6 个类别的生成质量
- 弱类别（C/D）：改进版是否明显更好

---

## 效果不好怎么办？

### 情况 1：输出全是乱码或重复

**原因**：过拟合。
**解决**：`python train_hecaitou.py --epochs 1`

### 情况 2：风格不够像

**原因**：训练不足。
**解决**：`python train_hecaitou.py --epochs 3 --lora-r 32`

### 情况 3：训练中途中断

**解决**：`python train_hecaitou.py --resume`

### 情况 4：显存不足 (CUDA OOM)

**解决**：

```powershell
python train_hecaitou.py --max-seq-len 2048 --lora-r 8
```

### 情况 5：某个类别生成质量差

**解决**：对少数类别过采样重新训练：

```powershell
python prepare_training_data.py --oversample
python train_hecaitou.py
```

---

## 常见问题

### Q: 依赖版本冲突怎么办？

```powershell
conda activate unsloth_env
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
```

关键原则：**不要单独 pip install trl 或 datasets**。

### Q: 模型下载太慢？

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python train_hecaitou.py
```

### Q: v2.0 的旧数据还能用吗？

能用。训练脚本会自动检测旧版 messages 格式并转换为纯文本继续训练。但建议重新生成 v4.0 数据以获得 completion-only loss 优势。

```powershell
# 迁移步骤
python prepare_training_data.py              # 重新生成 v4.0 数据
python train_hecaitou.py                     # 用新数据训练
```

### Q: 推理时需要按什么格式输入？

推理时**必须提供与训练一致的格式**，否则分布不匹配会导致生成质量下降。三种格式都可以：

```
# 最简（模板 C）
[社会观察] 写一篇关于「XXX」的文章

# 中等（模板 B）
写一篇社会观察类的文章，题目是「XXX」。
写作方向：社会现象的冷静剖析

# 完整（模板 A）
请以和菜头的风格，写一篇社会观察类文章。
类别：社会观察
标题：XXX
概要：简要描述文章主旨...
核心意向：社会现象的冷静剖析
```

### Q: 以后和菜头写了新文章，怎么更新？

1. 增量爬取：`python crawler.py --month 2026-04`
2. 运行分类：`python classify_articles.py`
3. 重新生成训练数据：`python prepare_training_data.py`
4. 重新训练：`python train_hecaitou.py`

---

## 附录 A：路线 B — Google Colab 云端训练

> 如果你没有 NVIDIA 显卡，使用 Colab 的免费 T4 GPU。

### Step 2B：打开 Colab 并设置 GPU

1. 访问 https://colab.research.google.com
2. 新建笔记本 → "运行时 → 更改运行时类型 → GPU（T4）"

### Step 3B：在 Colab 中运行以下代码块

#### 代码块 1/5：安装 Unsloth（3-5 分钟）

```python
!pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo 2>&1 | tail -5
import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
```

#### 代码块 2/5：上传并加载数据（1 分钟）

```python
from google.colab import files
from datasets import load_dataset
import os

os.makedirs('/content/data', exist_ok=True)

print('请选择 train.jsonl 文件：')
uploaded = files.upload()
for name, data in uploaded.items():
    open(f'/content/data/{name}', 'wb').write(data)

print('请选择 val.jsonl 文件：')
uploaded = files.upload()
for name, data in uploaded.items():
    open(f'/content/data/{name}', 'wb').write(data)

train_dataset = load_dataset('json', data_files='/content/data/train.jsonl', split='train')
val_dataset = load_dataset('json', data_files='/content/data/val.jsonl', split='train')
print(f'训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条')
```

#### 代码块 3/5：加载模型 + LoRA（5-10 分钟）

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=2048,
)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total/1e9:.2f}B, 可训练: {trainable/1e6:.1f}M ({100*trainable/total:.2f}%)")
```

#### 代码块 4/5：训练（1-3 小时）

```python
import time
from trl import SFTTrainer, SFTConfig

# prompt/completion 纯文本格式 + completion_only_loss=True
# 不需要 formatting_func 或 DataCollatorForCompletionOnlyLM
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir="/content/hecaitou_output",
        max_seq_length=2048,
        completion_only_loss=True,  # 仅 completion 参与 loss
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        fp16=False,
        bf16=False,
        seed=42,
        report_to="none",
    ),
)

start = time.time()
stats = trainer.train()
elapsed = (time.time() - start) / 60
print(f"\n训练完成！步数: {stats.global_step}, Loss: {stats.training_loss:.4f}, 用时: {elapsed:.0f} 分钟")
```

#### 代码块 5/5：导出 GGUF 并下载

```python
import glob, os
from google.colab import files

print("导出 GGUF（Q4_K_M 量化）...")
model.save_pretrained_gguf("/content/hecaitou_gguf", tokenizer, quantization_method="q4_k_m")

model.save_pretrained("/content/lora_adapter")
tokenizer.save_pretrained("/content/lora_adapter")

for f in glob.glob("/content/hecaitou_gguf/*.gguf"):
    size_gb = os.path.getsize(f) / 1024**3
    print(f"  {os.path.basename(f)} ({size_gb:.1f} GB)")
    files.download(f)
```

---

## 附录 B：训练参数速查表

| 参数 | 8GB GPU | 12GB GPU | 16GB+ GPU | 说明 |
|------|---------|----------|-----------|------|
| 模型 | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-4B/9B | bf16 LoRA |
| LoRA r | 16 | 16 | 16-32 | 越大学习力越强 |
| LoRA alpha | 16 | 16 | 16-32 | 通常 = r |
| max_seq_len | 4096 | 4096 | 2048-4096 | 覆盖 95%+ 文章 |
| epochs | 2 | 2 | 2 | 过拟合则用 1 |
| batch (effective) | 4 | 4 | 4 | batch_size=1, grad_accum=4 |
| weight_decay | 0.01 | 0.01 | 0.01 | 轻微正则 |

---

## 附录 C：数据格式说明

### v4.2 prompt/completion 格式（推荐）

prompt 和 completion 均为**纯自然语言文本**，不含任何 ChatML 特殊标记：

```json
{
  "prompt": "你是和菜头，运营公众号「槽边往事」。写作风格：温和的刻薄，冷幽默...\n\n请以和菜头的风格，写一篇社会观察类文章。\n类别：社会观察\n标题：XXX\n概要：...\n核心意向：...",
  "completion": "# XXX\n\n正文..."
}
```

**关键点**：
- prompt 是 system 指令 + user 请求的纯文本拼接
- completion 是文章正文的纯文本
- **不含** `<|im_start|>`、`<|im_end|>` 等特殊 token
- TRL 通过 `SFTConfig(completion_only_loss=True)` 自动处理：
  1. 用 `tokenizer.apply_chat_template()` 添加特殊 token
  2. 将 prompt 对应的 label 设为 -100（不参与梯度）
  3. 仅 completion 部分参与 loss 计算

> **重要**：如果你的旧数据包含 ChatML 特殊标记或 list[dict] 格式，
> 请重新运行 `python prepare_training_data.py` 生成新数据。

### 旧版 messages 格式（向后兼容）

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

训练脚本会自动检测并转换为纯文本格式（全序列 loss）。建议重新生成 v4.2 数据以获得 completion-only loss 优势。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `prepare_training_data.py` | 生成训练数据 v4.0（分层切分 + 混合模板） |
| `train_hecaitou.py` | 训练脚本 v4.0（completion-only loss + 实验管理） |
| `training_data/train.jsonl` | 训练集 80%（运行 prepare 后生成） |
| `training_data/val.jsonl` | 验证集 10% |
| `training_data/test.jsonl` | 测试集 10% |
| `training_data/data_report.txt` | 数据集详细统计报告 |
| `FINETUNE_GUIDE.md` | 本文档 |

---

*版本 v3.2 | 2026-03-13 | 修复 completion-only loss 兼容性 + 全面升级数据管线 + 训练脚本*
