# 和菜头风格 LoRA 微调：从零到部署完整指南 v2.0

> **更新 2026-03-11**：新增 Windows 本地训练完整步骤（RTX 5060 等），修复 Colab 上的所有已知问题
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
   完成！
```

---

## Step 0：5 分钟理解原理（可跳过）

### 什么是 LoRA 微调？

你现在的方案是：用 System Prompt（1000 字规则）告诉通用模型"请像和菜头那样写"。这就像给一个外国人一张中餐菜谱，让他照着做——能做出来，但味道差很远。

**LoRA 微调**是让模型直接读 1062 篇和菜头的文章，从参数层面学会他的写作节奏。这就像让那个外国人在中餐馆当了三个月学徒——他不需要看菜谱了，因为手感已经在肌肉里了。

### GPU 显存 vs 模型选择

| GPU | 显存 | 推荐模型 | 训练时间 |
|-----|------|----------|---------|
| RTX 5060 / 4060 | 8 GB | Qwen3.5-4B | 1-2 小时 |
| RTX 3060 12G / 4070 | 12 GB | Qwen3.5-4B 或 9B | 1-3 小时 |
| RTX 4090 / A6000 | 24 GB | Qwen3.5-9B | 2-3 小时 |
| Tesla T4 (Colab) | 16 GB | Qwen3.5-4B（推荐）或 9B | 1-3 小时 |
| 无 NVIDIA GPU | - | 用 Colab 路线 B | 见附录 |

> 4B 微调后效果 >> 35B MoE + prompt engineering。不用纠结模型大小。

---

## Step 1：本地生成训练数据（2 分钟）

### 1.1 打开 PowerShell，进入项目目录

```powershell
cd C:\Users\hzjin\hecaitou-blog-archive
```

### 1.2 拉取最新代码

```powershell
git pull origin main
```

### 1.3 运行数据准备脚本

```powershell
python prepare_training_data.py
```

你会看到：

```
[Step 1] 加载分类文章...
  A_社会观察.md: 195 篇
  B_技术产品.md: 219 篇
  ...
  共加载 1067 篇文章

[Step 3] 生成训练样本...
  总样本数: 1062

[Step 4] 划分数据集...
  训练集: 956 条
  验证集: 106 条

[Step 5] 文件已写入:
  training_data/train.jsonl  (5.6 MB)
  training_data/eval.jsonl  (673 KB)
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

你的命令行提示符前面会出现 `(unsloth_env)`，表示环境已激活。

#### 2A.3 安装 PyTorch（匹配你的 CUDA 版本）

先确认 CUDA 版本：

```powershell
nvidia-smi
```

看右上角的 "CUDA Version"。你的 RTX 5060 显示 CUDA 12.9，安装对应版本：

```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 如果你的 CUDA 版本是 13.0，也用 `cu128`（PyTorch 向下兼容）。
> 如果是 CUDA 12.1 或 12.4，改成 `cu121` 或 `cu124`。

验证 PyTorch 能看到 GPU：

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB')"
```

预期输出：`CUDA: True, GPU: NVIDIA GeForce RTX 5060 ..., VRAM: 8.0GB`

**如果显示 `CUDA: False`，PyTorch 安装有问题，不要继续。**重新运行上面的 pip 命令。

#### 2A.4 安装 Unsloth

```powershell
pip install unsloth
```

> **重要**：不要再单独运行 `pip install --upgrade trl datasets`！
> Unsloth 已经自带了兼容版本的 trl 和 datasets。
> 单独升级会导致版本冲突（你之前遇到的 `unsloth_zoo requires trl<=0.24.0 but trl 0.29.0 was installed` 就是这个原因）。

#### 2A.5 验证安装

```powershell
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

看到 `Unsloth OK` 就说明安装成功。如果报错，尝试：

```powershell
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
```

### Step 3A：开始训练（一条命令）

```powershell
cd C:\Users\hzjin\hecaitou-blog-archive
conda activate unsloth_env
python train_hecaitou.py
```

脚本会自动：
1. 检测你的 GPU（RTX 5060, 8GB）
2. 自动选择 Qwen3.5-4B 模型（适合 8GB 显存）
3. 自动配置 LoRA r=16, max_seq_len=1024
4. 下载模型（首次约 8GB，需 10-30 分钟取决于网速）
5. 开始训练（约 1-2 小时）

你会看到：

```
[自动配置] 模型: Qwen/Qwen3.5-4B
[自动配置] LoRA r: 16
[自动配置] 最大序列长度: 1024
...
[数据] 训练集: 956 条
[数据] 验证集: 106 条
...
开始训练... (Ctrl+C 可中断，下次用 --resume 恢复)

Step 10: loss = 2.1234
Step 20: loss = 1.8765
...
```

**loss 持续下降 = 正在学习**。正常的最终 loss 大约在 0.8-1.2。

> **训练过程中可以干别的**，不需要盯着。但不要关闭 Anaconda Powershell 窗口。
> 如果意外中断，用 `python train_hecaitou.py --resume` 从断点恢复。

#### 自定义参数（可选）

```powershell
# 如果有 12GB+ 显存，可以用 9B 模型
python train_hecaitou.py --model Qwen/Qwen3.5-9B --lora-r 32 --max-seq-len 2048

# 减少训练轮次（更快但效果稍差）
python train_hecaitou.py --epochs 2

# 增加 LoRA 秩（效果更好但更慢、更耗显存）
python train_hecaitou.py --lora-r 32
```

### Step 4A：导出并导入 Ollama

训练完成后，脚本会自动导出 GGUF 文件到 `hecaitou_output/gguf/`。

> **注意**：Windows 上 GGUF 导出可能会失败。如果看到导出失败提示，使用下面的 LoRA 适配器方案替代。

#### 方案一：GGUF 导出成功

1. 进入导出目录：

```powershell
dir hecaitou_output\gguf\
```

2. 在该目录创建 `Modelfile` 文本文件：

```text
FROM ./hecaitou_gguf-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER num_ctx 16384
SYSTEM "你是和菜头，运营公众号「槽边往事」。"
```

> `FROM` 后面的文件名要和实际的 `.gguf` 文件名一致。

3. 导入 Ollama：

```powershell
cd hecaitou_output\gguf
ollama create hecaitou-writer -f Modelfile
```

#### 方案二：GGUF 导出失败（用 LoRA 适配器）

LoRA 适配器已经保存在 `hecaitou_output/lora_adapter/`。你可以通过以下方式使用：

1. 在 Colab（免费）上仅运行导出步骤：

```python
from unsloth import FastLanguageModel

# 上传 lora_adapter 文件夹到 Colab
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/lora_adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)
model.save_pretrained_gguf("/content/hecaitou_gguf", tokenizer, quantization_method="q4_k_m")

# 下载 GGUF 文件
from google.colab import files
import glob
for f in glob.glob("/content/hecaitou_gguf/*.gguf"):
    files.download(f)
```

2. 或者安装 llama.cpp 在本地转换（高级用户）。

### Step 5：测试效果

```powershell
ollama run hecaitou-writer "写一篇关于'最近大家都在用AI写东西'的文章"
```

如果模型输出了一篇带有和菜头风格的文章（冷幽默、短句、自嘲、第一人称"我"），微调成功！

#### 切换 hecaitou_writer.py 的模型

编辑 `hecaitou_writer.py`，找到配置部分，修改：

```python
_config = {
    ...
    "writer_model": "hecaitou-writer",      # 改成你的微调模型名
    "critic_model": "hecaitou-writer",      # 同上
    ...
}
```

然后运行：

```powershell
python hecaitou_writer.py --topic "最近大家都在用AI写东西" --quick
```

---

## 效果不好怎么办？

### 情况 1：输出全是乱码或重复

**原因**：训练轮次太多，过拟合了。
**解决**：`python train_hecaitou.py --epochs 1`

### 情况 2：风格不够像

**原因**：训练不足。
**解决**：增加 epochs 或 LoRA 秩：`python train_hecaitou.py --epochs 5 --lora-r 32`

### 情况 3：训练中途中断

**解决**：`python train_hecaitou.py --resume`

脚本会从最近的检查点（每 200 步自动保存）恢复。

### 情况 4：显存不足 (CUDA OOM)

**解决**：减小参数：

```powershell
python train_hecaitou.py --max-seq-len 512 --lora-r 8 --model Qwen/Qwen3.5-2B
```

---

## 常见问题

### Q: 依赖版本冲突怎么办？

最常见的原因是单独升级了 trl 或 datasets。解决方案：

```powershell
# 在 Conda 环境中重建
conda activate unsloth_env
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
```

关键原则：**不要单独 pip install trl 或 datasets**，让 unsloth 管理它们。

### Q: 模型下载太慢？

设置 HuggingFace 镜像（国内用户）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python train_hecaitou.py
```

或者提前手动下载：

```powershell
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./qwen35-4b
python train_hecaitou.py --model ./qwen35-4b
```

### Q: 训练完还需要 HECAITOU_STYLE_SKILLS.md 吗？

微调后模型从参数层面学会了风格，Skill 文件的作用大幅降低。但建议保留去 AI 化规则作为质量兜底。

### Q: 以后和菜头写了新文章，怎么更新？

1. 用爬虫增量爬取：`python crawler.py --month 2026-04`
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

创建 5 个代码单元格，依次粘贴并运行：

#### 代码块 1/5：安装 Unsloth（3-5 分钟）

```python
!pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo 2>&1 | tail -5
import torch
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
```

> 如果弹出"需要重启运行时"提示，点**取消**！重启会丢失已安装的库。

#### 代码块 2/5：上传并加载数据（1 分钟）

```python
from google.colab import files
from datasets import load_dataset
import os, json

os.makedirs('/content/data', exist_ok=True)

print('请选择 train.jsonl 文件：')
uploaded = files.upload()
for name, data in uploaded.items():
    open(f'/content/data/{name}', 'wb').write(data)

print('请选择 eval.jsonl 文件：')
uploaded = files.upload()
for name, data in uploaded.items():
    open(f'/content/data/{name}', 'wb').write(data)

train_dataset = load_dataset('json', data_files='/content/data/train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='/content/data/eval.jsonl', split='train')
print(f'训练集: {len(train_dataset)} 条')
print(f'验证集: {len(eval_dataset)} 条')
```

#### 代码块 3/5：加载模型 + LoRA（5-10 分钟，含下载）

```python
from unsloth import FastLanguageModel
import torch

# 使用 4B 模型（下载快，训练快，效果好）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=2048,
    load_in_4bit=True,
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
print(f"显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
```

#### 代码块 4/5：预处理数据 + 开始训练（1-3 小时）

```python
import torch, json, time
from trl import SFTTrainer, SFTConfig

# ===== 预转换数据为纯文本（修复 Unsloth 兼容性问题） =====
def convert_to_text(example):
    messages = example["messages"]
    parts = []
    for msg in messages:
        if isinstance(msg, str):
            try:
                msg = json.loads(msg)
            except:
                continue
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return {"text": "\n".join(parts)}

train_text = train_dataset.map(convert_to_text, remove_columns=train_dataset.column_names)
eval_text = eval_dataset.map(convert_to_text, remove_columns=eval_dataset.column_names)
print(f"数据转换完成：训练 {len(train_text)} 条，验证 {len(eval_text)} 条")
print(f"预览：{train_text[0]['text'][:150]}...")

# ===== 训练 =====
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_text,
    eval_dataset=eval_text,
    args=SFTConfig(
        output_dir="/content/hecaitou_output",
        max_seq_length=2048,
        dataset_text_field="text",
        num_train_epochs=3,
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

total_steps = len(train_text) * 3 // 4
print(f"训练计划：{len(train_text)} 样本 x 3 轮 / batch 4 = ~{total_steps} 步")
print(f"预计用时：{total_steps*4//60}-{total_steps*6//60} 分钟\n")

start = time.time()
stats = trainer.train()
elapsed = (time.time() - start) / 60
print(f"\n训练完成！步数: {stats.global_step}, Loss: {stats.training_loss:.4f}, 用时: {elapsed:.0f} 分钟")
```

#### 代码块 5/5：导出 GGUF 并下载（10-15 分钟）

```python
import glob, os
from google.colab import files

print("导出 GGUF（Q4_K_M 量化）... 约 10-15 分钟")
model.save_pretrained_gguf("/content/hecaitou_gguf", tokenizer, quantization_method="q4_k_m")

# 保存 LoRA 适配器（体积小，备份用）
model.save_pretrained("/content/lora_adapter")
tokenizer.save_pretrained("/content/lora_adapter")

# 下载 GGUF 到本地
gguf_files = glob.glob("/content/hecaitou_gguf/*.gguf")
for f in gguf_files:
    size_gb = os.path.getsize(f) / 1024**3
    print(f"  {os.path.basename(f)} ({size_gb:.1f} GB)")
    print(f"  正在下载到你的电脑...")
    files.download(f)

print("\n全部完成！下载的 .gguf 文件就是你的微调模型。")
print("接下来参考 Step 4A 导入 Ollama。")
```

### Step 4B：导入 Ollama

同 Step 4A 的方案一。

---

## 附录 B：技术细节

### 为什么预转换为纯文本？

Unsloth 的 SFTTrainer 在处理 ChatML `messages` 格式时存在多个已知 bug：
- `formatting_func` 在单样本测试时返回空列表 → `IndexError`
- 部分版本将 messages 中的 dict 序列化为字符串 → `JSONDecodeError`
- `AttributeError: 'str' object has no attribute 'get'`

我们的解决方案是在训练前将 messages 格式手动转换为 ChatML 纯文本：

```
<|im_start|>system
你是和菜头...
<|im_end|>
<|im_start|>user
写一篇关于...
<|im_end|>
<|im_start|>assistant
# 标题
正文...
<|im_end|>
```

然后使用 `dataset_text_field="text"` 直接训练，完全绕过了 Unsloth 的 messages 解析逻辑。

### 训练参数速查表

| 参数 | 8GB GPU | 12GB GPU | 16GB+ GPU | 说明 |
|------|---------|----------|-----------|------|
| 模型 | Qwen3.5-4B | Qwen3.5-4B/9B | Qwen3.5-9B | 越大效果越好 |
| LoRA r | 16 | 16-32 | 32 | 越大学习能力越强 |
| max_seq_len | 1024 | 1024-2048 | 2048 | 覆盖文章长度 |
| batch_size | 1 | 1 | 1-2 | 一般不需要改 |
| grad_accum | 4 | 4 | 4 | 等效 batch=4 |
| epochs | 3 | 3 | 3 | 过拟合则减小 |

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `prepare_training_data.py` | 生成训练数据（本地运行） |
| `train_hecaitou.py` | 本地训练脚本（自动检测 GPU 并配置） |
| `training_data/train.jsonl` | 训练集（运行 prepare 后生成） |
| `training_data/eval.jsonl` | 验证集（运行 prepare 后生成） |
| `FINETUNE_GUIDE.md` | 本文档 |

---

*版本 v2.0 | 2026-03-11 | 新增 Windows 本地训练 + 修复 Colab 兼容性问题*
