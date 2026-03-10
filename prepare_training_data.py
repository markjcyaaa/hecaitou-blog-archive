#!/usr/bin/env python3
"""
和菜头语料 → Unsloth SFT 训练数据 转换脚本
==========================================
从 classified/ 目录读取 1073 篇文章，转换为 Unsloth/TRL 可用的
ChatML 格式训练数据（JSONL），直接用于 LoRA 微调。

用法：
  python prepare_training_data.py                    # 默认输出到 training_data/
  python prepare_training_data.py --output ./data    # 指定输出目录
  python prepare_training_data.py --preview 5        # 只预览前5条，不写文件
  python prepare_training_data.py --max-length 3000  # 限制单篇最大字符数

输出文件：
  training_data/
    ├── train.jsonl          # 训练集（90%）
    ├── eval.jsonl           # 验证集（10%）
    └── data_report.txt      # 数据集统计报告
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
CLASSIFIED_DIR = SCRIPT_DIR / "classified"

# 分类文件映射
CLASSIFIED_FILES = {
    "A": ("A_社会观察.md", "社会观察"),
    "B": ("B_技术产品.md", "技术产品评论"),
    "C": ("C_生死无常.md", "生死无常感悟"),
    "D": ("D_自省修行.md", "自省修行"),
    "E": ("E_文化阅读.md", "文化阅读评论"),
    "F": ("F_日常生活.md", "日常生活随笔"),
}

# 系统提示（微调时嵌入模型，推理时只需极简提示即可触发风格）
SYSTEM_PROMPT = "你是和菜头，运营公众号「槽边往事」。写作风格：温和的刻薄，冷幽默，短句为主，善用自嘲和比喻，第一人称，结尾留余味。"

# ============================================================
# 文章解析
# ============================================================

def parse_articles_from_file(filepath: Path, category_code: str) -> List[dict]:
    """从单个分类 Markdown 文件中解析出所有文章。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"  [跳过] 无法读取 {filepath}: {e}")
        return []

    articles = []
    parts = re.split(r'\n### ', content)

    for part in parts[1:]:
        lines = part.strip().split("\n")
        title = lines[0].strip()
        if not title or title.startswith("目录"):
            continue

        body = "\n".join(lines[1:]).strip()

        # 提取日期
        date_match = re.search(r'\*\*日期\*\*[：:]\s*(\d{4}-\d{2}-\d{2})', body)
        date = date_match.group(1) if date_match else ""

        # 清理：去掉元数据行和分隔线
        clean_body = body
        clean_body = re.sub(r'\*\*日期\*\*[：:].*?\n', '', clean_body)
        clean_body = re.sub(r'\*\*原文链接\*\*[：:].*?\n', '', clean_body)
        clean_body = re.sub(r'^---+\s*$', '', clean_body, flags=re.MULTILINE)

        # 去掉图片 Markdown（模型学不到图片）
        clean_body = re.sub(r'!\[.*?\]\(.*?\)', '', clean_body)
        # 去掉空的链接图片
        clean_body = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', clean_body)
        # 保留文字链接的文字部分
        clean_body = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_body)

        # 清理多余空行
        clean_body = re.sub(r'\n{3,}', '\n\n', clean_body).strip()

        if len(clean_body) < 100:
            continue

        articles.append({
            "title": title,
            "body": clean_body,
            "category": category_code,
            "date": date,
            "char_count": len(clean_body),
        })

    return articles


def load_all_articles() -> List[dict]:
    """加载全部分类文章。"""
    all_articles = []
    for code, (fname, _label) in CLASSIFIED_FILES.items():
        fpath = CLASSIFIED_DIR / fname
        if not fpath.exists():
            print(f"  [跳过] 文件不存在: {fpath}")
            continue
        articles = parse_articles_from_file(fpath, code)
        print(f"  {fname}: {len(articles)} 篇")
        all_articles.extend(articles)
    return all_articles


# ============================================================
# 指令生成
# ============================================================

def generate_instruction(article: dict) -> str:
    """根据文章的标题和类别，生成自然的写作指令。"""
    title = article["title"]
    cat_code = article["category"]
    _, cat_label = CLASSIFIED_FILES.get(cat_code, ("", ""))

    # 多种指令模板，增加多样性
    templates = [
        f"写一篇关于「{title}」的文章",
        f"以「{title}」为题写一篇文章",
        f"请写一篇文章，主题是：{title}",
        f"用你的风格写一篇「{title}」",
        f"今天聊聊「{title}」这个话题",
    ]

    # 带类别的模板（30% 概率使用）
    category_templates = [
        f"写一篇{cat_label}类的文章，题目是「{title}」",
        f"从{cat_label}的角度，写一篇关于「{title}」的文章",
    ]

    if random.random() < 0.3 and cat_label:
        return random.choice(category_templates)
    else:
        return random.choice(templates)


# ============================================================
# 转换为训练格式
# ============================================================

def article_to_training_sample(article: dict) -> dict:
    """将单篇文章转换为 ChatML 训练样本。"""
    instruction = generate_instruction(article)

    # 输出格式：标题 + 正文（和 hecaitou_writer.py 的输出格式一致）
    output_text = f"# {article['title']}\n\n{article['body']}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output_text},
        ],
        # 元数据（不参与训练，方便调试）
        "_meta": {
            "title": article["title"],
            "category": article["category"],
            "date": article["date"],
            "char_count": article["char_count"],
        }
    }


def estimate_tokens(text: str) -> int:
    """粗略估算中文 token 数（1 中文字 ≈ 1.5 tokens）。"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars * 0.3)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="将和菜头语料转换为 LoRA 微调训练数据")
    parser.add_argument("--output", default="training_data", help="输出目录（默认 training_data/）")
    parser.add_argument("--preview", type=int, default=0, help="只预览前 N 条，不写文件")
    parser.add_argument("--max-length", type=int, default=4000, help="单篇最大字符数（超长文章截断）")
    parser.add_argument("--min-length", type=int, default=200, help="单篇最小字符数（太短的跳过）")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="验证集比例（默认 10%%）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("和菜头语料 → LoRA 训练数据 转换工具")
    print("=" * 60)

    # 1. 加载文章
    print("\n[Step 1] 加载分类文章...")
    articles = load_all_articles()
    print(f"  共加载 {len(articles)} 篇文章")

    # 2. 过滤
    print(f"\n[Step 2] 过滤（{args.min_length} ≤ 字数 ≤ {args.max_length}）...")
    filtered = []
    skipped_short = 0
    truncated = 0
    for a in articles:
        if a["char_count"] < args.min_length:
            skipped_short += 1
            continue
        if a["char_count"] > args.max_length:
            a["body"] = a["body"][:args.max_length]
            a["char_count"] = args.max_length
            truncated += 1
        filtered.append(a)
    print(f"  保留 {len(filtered)} 篇 | 跳过过短 {skipped_short} 篇 | 截断过长 {truncated} 篇")

    # 3. 转换为训练格式
    print("\n[Step 3] 生成训练样本...")
    samples = [article_to_training_sample(a) for a in filtered]

    # 统计
    total_tokens = 0
    token_counts = []
    for s in samples:
        full_text = s["messages"][0]["content"] + s["messages"][1]["content"] + s["messages"][2]["content"]
        tokens = estimate_tokens(full_text)
        total_tokens += tokens
        token_counts.append(tokens)
    token_counts.sort()

    print(f"  总样本数: {len(samples)}")
    print(f"  预估总 tokens: {total_tokens:,}")
    print(f"  平均 tokens/样本: {total_tokens // len(samples)}")
    print(f"  中位数 tokens: {token_counts[len(token_counts)//2]}")
    print(f"  最大 tokens: {token_counts[-1]}  最小: {token_counts[0]}")

    # 4. 预览模式
    if args.preview > 0:
        print(f"\n[预览模式] 显示前 {args.preview} 条：")
        for i, s in enumerate(samples[:args.preview]):
            meta = s["_meta"]
            print(f"\n--- 样本 {i+1} ---")
            print(f"标题: {meta['title']}  类别: {meta['category']}  字数: {meta['char_count']}")
            print(f"System: {s['messages'][0]['content'][:80]}...")
            print(f"User: {s['messages'][1]['content']}")
            print(f"Assistant: {s['messages'][2]['content'][:150]}...")
        return

    # 5. 划分训练/验证集
    print(f"\n[Step 4] 划分数据集（验证集比例 {args.eval_ratio:.0%}）...")
    random.shuffle(samples)
    eval_count = max(1, int(len(samples) * args.eval_ratio))
    eval_samples = samples[:eval_count]
    train_samples = samples[eval_count:]
    print(f"  训练集: {len(train_samples)} 条")
    print(f"  验证集: {len(eval_samples)} 条")

    # 6. 写入文件
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"

    def write_jsonl(filepath, data):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                # 写入时去掉 _meta 字段
                clean = {"messages": item["messages"]}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    write_jsonl(train_file, train_samples)
    write_jsonl(eval_file, eval_samples)

    print(f"\n[Step 5] 文件已写入:")
    print(f"  {train_file}  ({os.path.getsize(train_file) / 1024 / 1024:.1f} MB)")
    print(f"  {eval_file}  ({os.path.getsize(eval_file) / 1024:.0f} KB)")

    # 7. 写入统计报告
    report_file = output_dir / "data_report.txt"
    cat_counts = {}
    for s in samples:
        cat = s["_meta"]["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("和菜头 LoRA 微调数据集报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"总样本数: {len(samples)}\n")
        f.write(f"训练集: {len(train_samples)} 条\n")
        f.write(f"验证集: {len(eval_samples)} 条\n")
        f.write(f"预估总 tokens: {total_tokens:,}\n")
        f.write(f"平均 tokens/样本: {total_tokens // len(samples)}\n")
        f.write(f"字符限制: {args.min_length}-{args.max_length}\n\n")
        f.write("各类别分布:\n")
        for code in sorted(cat_counts.keys()):
            _, label = CLASSIFIED_FILES.get(code, ("", "未知"))
            f.write(f"  {code} {label}: {cat_counts[code]} 篇\n")
        f.write(f"\nSystem Prompt:\n  {SYSTEM_PROMPT}\n")

    print(f"  {report_file}")

    # 8. 完成提示
    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print()
    print("下一步：将 training_data/ 上传到 Google Colab 进行训练")
    print("详细步骤请看 FINETUNE_GUIDE.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
