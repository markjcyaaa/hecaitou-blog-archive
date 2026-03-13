#!/usr/bin/env python3
"""
和菜头语料 → Unsloth SFT 训练数据 转换脚本 v3.0
==============================================
从 classified/ 目录读取 1073 篇文章，转换为 Unsloth/TRL 可用的
prompt-completion 格式训练数据（JSONL），支持 completion-only loss。

v3.0 更新（2026-03-13）：
  - 新增「增强 prompt」模式：自动提取概要 + 核心意向，提供更强语义锚点
  - 切换为 prompt/completion 格式（TRL 原生支持 completion-only loss）
  - 概要由规则自动提取（无需外部 API），保持 2-3 句、点到核心主题
  - 保留旧版 messages 格式作为备选（--format messages）

用法：
  python prepare_training_data.py                         # 增强格式（推荐）
  python prepare_training_data.py --format messages       # 旧版 ChatML 格式
  python prepare_training_data.py --preview 5             # 预览前5条
  python prepare_training_data.py --max-length 6000       # 放宽长度限制
  python prepare_training_data.py --output ./data         # 指定输出目录

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

# 分类到写作意向的映射
CATEGORY_INTENTIONS = {
    "A": ["社会现象的冷静剖析", "对公共事件的个人解读", "温和而尖锐的社会批评"],
    "B": ["产品体验中的生活哲学", "技术背后的人性观察", "消费主义的反思"],
    "C": ["面对生死的坦然", "无常中的珍重", "告别与释然"],
    "D": ["日常修行的体悟", "自我审视的诚实", "中年心态的转变"],
    "E": ["阅读触发的思考", "文化现象的品评", "书影音中的人生映照"],
    "F": ["日常琐事中的趣味", "生活细节的诗意发现", "平凡中的温暖与自嘲"],
}

# 系统提示
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
# 概要自动提取（规则式，无需外部 API）
# ============================================================

def extract_summary(article: dict) -> str:
    """
    从文章正文中自动提取概要（2-3 句话）。
    
    策略：
    1. 取前 2-3 个段落的首句（段落开头往往是核心论点）
    2. 如果文章有明显的转折句（然而、但是、不过），也提取
    3. 控制在 80-150 字之间
    """
    body = article["body"]
    title = article["title"]
    
    # 按段落拆分（双换行）
    paragraphs = [p.strip() for p in re.split(r'\n\n+', body) if p.strip()]
    
    if not paragraphs:
        return f"一篇关于「{title}」的文章。"
    
    summary_parts = []
    
    # 策略1：取前几个段落的首句
    for i, para in enumerate(paragraphs[:4]):
        # 提取首句（中文句号、问号、感叹号结尾）
        sentence_match = re.match(r'(.+?[。？！…])', para)
        if sentence_match:
            sentence = sentence_match.group(1)
            # 跳过太短的句子（< 10 字）
            if len(sentence) >= 10:
                summary_parts.append(sentence)
                if len("".join(summary_parts)) >= 80:
                    break
    
    # 如果首句提取不够，用段落前 N 个字
    if len(summary_parts) < 2 and paragraphs:
        first_para = paragraphs[0]
        if len(first_para) > 100:
            summary_parts = [first_para[:100] + "……"]
        else:
            summary_parts = [first_para]
    
    # 策略2：找转折句（如果前面已经够了就跳过）
    if len("".join(summary_parts)) < 60:
        for para in paragraphs[2:6]:
            turn_match = re.search(r'((?:然而|但是|不过|可是|问题在于|关键是).+?[。？！])', para)
            if turn_match:
                summary_parts.append(turn_match.group(1))
                break
    
    summary = "".join(summary_parts)
    
    # 控制长度：截断到 150 字
    if len(summary) > 150:
        # 在句号处截断
        cut_pos = summary.rfind("。", 0, 150)
        if cut_pos > 50:
            summary = summary[:cut_pos + 1]
        else:
            summary = summary[:150] + "……"
    
    return summary


def extract_intention(article: dict) -> str:
    """
    根据文章类别和内容，提取核心写作意向。
    
    策略：从预定义的意向池中选择，并结合标题关键词。
    """
    cat_code = article["category"]
    title = article["title"]
    
    # 从分类对应的意向池中选 1-2 个
    intentions = CATEGORY_INTENTIONS.get(cat_code, ["个人随笔"])
    selected = random.sample(intentions, min(2, len(intentions)))
    
    # 基于标题关键词补充意向
    title_keywords = {
        "死|去世|告别|离开|悼": "对逝者的怀念与生命的感悟",
        "AI|人工智能|GPT|算法": "技术浪潮下的人文思考",
        "读|书|影|片|电影": "作品触发的生活联想",
        "吃|喝|食|茶|酒|咖啡": "食物中的生活美学",
        "旅|行|路|城市": "旅途中的内心独白",
        "年|老|中年|回忆": "时光流逝的坦然面对",
    }
    
    for pattern, intention in title_keywords.items():
        if re.search(pattern, title):
            selected.append(intention)
            break
    
    return "、".join(selected[:3])


# ============================================================
# 指令生成
# ============================================================

def generate_instruction_simple(article: dict) -> str:
    """旧版：简单指令（仅标题）。"""
    title = article["title"]
    cat_code = article["category"]
    _, cat_label = CLASSIFIED_FILES.get(cat_code, ("", ""))

    templates = [
        f"写一篇关于「{title}」的文章",
        f"以「{title}」为题写一篇文章",
        f"请写一篇文章，主题是：{title}",
        f"用你的风格写一篇「{title}」",
        f"今天聊聊「{title}」这个话题",
    ]

    category_templates = [
        f"写一篇{cat_label}类的文章，题目是「{title}」",
        f"从{cat_label}的角度，写一篇关于「{title}」的文章",
    ]

    if random.random() < 0.3 and cat_label:
        return random.choice(category_templates)
    else:
        return random.choice(templates)


def generate_instruction_enhanced(article: dict) -> str:
    """
    增强版：带概要和意向的丰富指令。
    
    格式：
      请以和菜头的风格写一篇文章。
      标题：{title}
      概要：{summary}
      核心意向：{intention}
    """
    title = article["title"]
    summary = extract_summary(article)
    intention = extract_intention(article)
    
    # 多种开头模板（增加多样性，但核心结构一致）
    openers = [
        "请以和菜头的风格写一篇文章。",
        "用和菜头「槽边往事」的笔触写一篇文章。",
        "以和菜头的口吻和风格，写以下文章。",
    ]
    
    opener = random.choice(openers)
    
    instruction = f"{opener}\n标题：{title}\n概要：{summary}\n核心意向：{intention}"
    
    return instruction


# ============================================================
# 转换为训练格式
# ============================================================

def article_to_prompt_completion(article: dict, enhanced: bool = True) -> dict:
    """
    将单篇文章转换为 prompt/completion 训练样本。
    
    TRL 的 prompt/completion 格式天然支持 completion-only loss：
    - prompt 部分不参与 loss 计算
    - completion 部分参与 loss 计算
    """
    if enhanced:
        instruction = generate_instruction_enhanced(article)
    else:
        instruction = generate_instruction_simple(article)
    
    # 构造 prompt（system + user，以 ChatML 对话格式）
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    
    # 构造 completion（assistant 回复）
    output_text = f"# {article['title']}\n\n{article['body']}"
    completion = [
        {"role": "assistant", "content": output_text},
    ]
    
    return {
        "prompt": prompt,
        "completion": completion,
        "_meta": {
            "title": article["title"],
            "category": article["category"],
            "date": article["date"],
            "char_count": article["char_count"],
        }
    }


def article_to_messages(article: dict, enhanced: bool = True) -> dict:
    """旧版：转换为 ChatML messages 格式（兼容旧训练脚本）。"""
    if enhanced:
        instruction = generate_instruction_enhanced(article)
    else:
        instruction = generate_instruction_simple(article)
    
    output_text = f"# {article['title']}\n\n{article['body']}"
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output_text},
        ],
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
    parser.add_argument("--max-length", type=int, default=6000, help="单篇最大字符数（默认 6000，覆盖 99%+ 文章）")
    parser.add_argument("--min-length", type=int, default=200, help="单篇最小字符数（太短的跳过）")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="验证集比例（默认 10%%）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--format", choices=["enhanced", "messages", "simple"],
                        default="enhanced",
                        help="数据格式：enhanced=增强prompt/completion（推荐），"
                             "messages=旧版ChatML messages，"
                             "simple=简单prompt无概要")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("和菜头语料 → LoRA 训练数据 转换工具 v3.0")
    print("=" * 60)
    print(f"  格式模式: {args.format}")

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
    
    use_enhanced = args.format == "enhanced"
    
    if args.format == "messages":
        samples = [article_to_messages(a, enhanced=False) for a in filtered]
        data_key = "messages"
    elif args.format == "simple":
        # prompt/completion 但用简单指令
        samples = [article_to_prompt_completion(a, enhanced=False) for a in filtered]
        data_key = "prompt"
    else:
        # enhanced: prompt/completion + 增强指令
        samples = [article_to_prompt_completion(a, enhanced=True) for a in filtered]
        data_key = "prompt"

    # 统计
    total_tokens = 0
    token_counts = []
    for s in samples:
        if data_key == "messages":
            full_text = "".join(m["content"] for m in s["messages"])
        else:
            prompt_text = "".join(m["content"] for m in s["prompt"])
            completion_text = "".join(m["content"] for m in s["completion"])
            full_text = prompt_text + completion_text
        tokens = estimate_tokens(full_text)
        total_tokens += tokens
        token_counts.append(tokens)
    token_counts.sort()

    print(f"  总样本数: {len(samples)}")
    print(f"  预估总 tokens: {total_tokens:,}")
    print(f"  平均 tokens/样本: {total_tokens // len(samples)}")
    print(f"  中位数 tokens: {token_counts[len(token_counts)//2]}")
    print(f"  最大 tokens: {token_counts[-1]}  最小: {token_counts[0]}")
    
    # 统计 token 分布
    for threshold in [1024, 2048, 3072, 4096, 5120, 6144]:
        count = sum(1 for t in token_counts if t <= threshold)
        pct = count / len(token_counts) * 100
        print(f"  ≤{threshold} tokens: {count} 条 ({pct:.1f}%)")

    # 4. 预览模式
    if args.preview > 0:
        print(f"\n[预览模式] 显示前 {args.preview} 条：")
        for i, s in enumerate(samples[:args.preview]):
            meta = s["_meta"]
            print(f"\n{'='*50}")
            print(f"样本 {i+1} | 标题: {meta['title']} | 类别: {meta['category']} | 字数: {meta['char_count']}")
            print("-" * 50)
            
            if data_key == "messages":
                for m in s["messages"]:
                    role = m["role"]
                    content = m["content"]
                    if role == "assistant":
                        content = content[:200] + "..."
                    print(f"[{role}] {content}")
            else:
                print("[PROMPT]")
                for m in s["prompt"]:
                    print(f"  [{m['role']}] {m['content']}")
                print("[COMPLETION]")
                for m in s["completion"]:
                    print(f"  [{m['role']}] {m['content'][:200]}...")
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
                clean = {k: v for k, v in item.items() if k != "_meta"}
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
        f.write("和菜头 LoRA 微调数据集报告 v3.0\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"数据格式: {args.format}\n")
        f.write(f"总样本数: {len(samples)}\n")
        f.write(f"训练集: {len(train_samples)} 条\n")
        f.write(f"验证集: {len(eval_samples)} 条\n")
        f.write(f"预估总 tokens: {total_tokens:,}\n")
        f.write(f"平均 tokens/样本: {total_tokens // len(samples)}\n")
        f.write(f"中位数 tokens: {token_counts[len(token_counts)//2]}\n")
        f.write(f"最大 tokens: {token_counts[-1]}\n")
        f.write(f"字符限制: {args.min_length}-{args.max_length}\n\n")
        
        f.write("Token 分布:\n")
        for threshold in [1024, 2048, 3072, 4096, 5120, 6144]:
            count = sum(1 for t in token_counts if t <= threshold)
            pct = count / len(token_counts) * 100
            f.write(f"  ≤{threshold}: {count} ({pct:.1f}%)\n")
        
        f.write("\n各类别分布:\n")
        for code in sorted(cat_counts.keys()):
            _, label = CLASSIFIED_FILES.get(code, ("", "未知"))
            f.write(f"  {code} {label}: {cat_counts[code]} 篇\n")
        
        f.write(f"\nSystem Prompt:\n  {SYSTEM_PROMPT}\n")
        
        if args.format == "enhanced":
            f.write(f"\n增强格式说明:\n")
            f.write(f"  - 使用 prompt/completion 格式（TRL 原生 completion-only loss）\n")
            f.write(f"  - prompt 包含：system 提示 + 增强指令（标题+概要+意向）\n")
            f.write(f"  - completion 包含：assistant 的完整文章\n")
            f.write(f"  - 训练时只在 completion 部分计算 loss\n")

    print(f"  {report_file}")

    # 8. 完成提示
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print()
    if args.format == "enhanced":
        print("数据格式: prompt/completion（增强版，含概要+意向）")
        print("训练特性: completion-only loss（只学写作，不学生成概要）")
        print()
    print("下一步：运行训练脚本")
    print("  python train_hecaitou.py")
    print()
    print("或指定参数：")
    print("  python train_hecaitou.py --model Qwen/Qwen3.5-2B --max-seq-len 4096")
    print("=" * 60)


if __name__ == "__main__":
    main()
