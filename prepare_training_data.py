#!/usr/bin/env python3
"""
和菜头语料 → Unsloth SFT 训练数据 转换脚本 v4.0
==============================================
从 classified/ 目录读取 ~1067 篇文章，转换为 Unsloth/TRL 可用的
prompt/completion 格式训练数据（JSONL），支持 completion-only loss。

v4.0 更新（2026-03-13）：
  - 文章级切分：整篇文章为一个样本，不拆段落
  - 分层 train/val/test 切分（80%/10%/10%），每个 split 均保持类别比例
  - 优先按时间排序再切分（>96% 文章有日期），无日期的随机分入
  - 3 种混合 prompt 模板（A 50%、B 30%、C 20%），均包含 category 字段
  - category 字段保留在 JSONL 中（_meta.category + prompt 文本内）
  - 少数类别（C/D）在训练集可选过采样（--oversample）
  - 重复标题 / 正文哈希去重
  - 详细统计日志：字符长度均值/P95、类别分布、去重报告、随机抽样展示
  - 支持 --baseline 生成无 category 的对照数据集

用法：
  python prepare_training_data.py                      # 默认（推荐）
  python prepare_training_data.py --preview 5          # 预览前5条
  python prepare_training_data.py --oversample         # 对少数类别 2x 过采样
  python prepare_training_data.py --baseline           # 生成无 category 的基线数据
  python prepare_training_data.py --format messages    # 旧版 ChatML 格式

输出文件：
  training_data/
    ├── train.jsonl          # 训练集（80%）
    ├── val.jsonl            # 验证集（10%）
    ├── test.jsonl           # 测试集（10%）
    └── data_report.txt      # 数据集统计报告
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
CLASSIFIED_DIR = SCRIPT_DIR / "classified"

# ============================================================
# 分类配置
# ============================================================

CLASSIFIED_FILES = {
    "A": ("A_社会观察.md", "社会观察"),
    "B": ("B_技术产品.md", "技术产品评论"),
    "C": ("C_生死无常.md", "生死无常感悟"),
    "D": ("D_自省修行.md", "自省修行"),
    "E": ("E_文化阅读.md", "文化阅读评论"),
    "F": ("F_日常生活.md", "日常生活随笔"),
}

CATEGORY_INTENTIONS = {
    "A": ["社会现象的冷静剖析", "对公共事件的个人解读", "温和而尖锐的社会批评"],
    "B": ["产品体验中的生活哲学", "技术背后的人性观察", "消费主义的反思"],
    "C": ["面对生死的坦然", "无常中的珍重", "告别与释然"],
    "D": ["日常修行的体悟", "自我审视的诚实", "中年心态的转变"],
    "E": ["阅读触发的思考", "文化现象的品评", "书影音中的人生映照"],
    "F": ["日常琐事中的趣味", "生活细节的诗意发现", "平凡中的温暖与自嘲"],
}

SYSTEM_PROMPT = "你是和菜头，运营公众号「槽边往事」。写作风格：温和的刻薄，冷幽默，短句为主，善用自嘲和比喻，第一人称，结尾留余味。"


# ============================================================
# 文章解析
# ============================================================

def parse_articles_from_file(filepath: Path, category_code: str) -> List[dict]:
    """从单个分类 Markdown 文件中解析出所有文章。"""
    try:
        content = filepath.read_text(encoding="utf-8")
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

        # 清理
        clean_body = body
        clean_body = re.sub(r'\*\*日期\*\*[：:].*?\n', '', clean_body)
        clean_body = re.sub(r'\*\*原文链接\*\*[：:].*?\n', '', clean_body)
        clean_body = re.sub(r'^---+\s*$', '', clean_body, flags=re.MULTILINE)
        clean_body = re.sub(r'!\[.*?\]\(.*?\)', '', clean_body)
        clean_body = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', clean_body)
        clean_body = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_body)
        clean_body = re.sub(r'\n{3,}', '\n\n', clean_body).strip()

        if len(clean_body) < 100:
            continue

        # 正文哈希（用于去重）
        body_hash = hashlib.md5(clean_body.encode("utf-8")).hexdigest()

        articles.append({
            "title": title,
            "body": clean_body,
            "category": category_code,
            "date": date,
            "char_count": len(clean_body),
            "body_hash": body_hash,
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
# 去重
# ============================================================

def deduplicate(articles: List[dict]) -> Tuple[List[dict], dict]:
    """
    去除重复文章（标题重复 或 正文哈希重复）。
    返回：(去重后文章列表, 去重统计)
    """
    seen_titles = {}
    seen_hashes = {}
    dedup_stats = {"dup_title": [], "dup_hash": []}
    unique = []

    for a in articles:
        title = a["title"]
        h = a["body_hash"]

        if title in seen_titles:
            dedup_stats["dup_title"].append(
                f"  「{title}」(cat={a['category']}) 与 (cat={seen_titles[title]}) 重复"
            )
            continue
        if h in seen_hashes:
            dedup_stats["dup_hash"].append(
                f"  hash={h[:8]}... 「{title}」(cat={a['category']}) "
                f"与 「{seen_hashes[h]}」 正文重复"
            )
            continue

        seen_titles[title] = a["category"]
        seen_hashes[h] = title
        unique.append(a)

    return unique, dedup_stats


# ============================================================
# 分层切分（文章级，优先时间排序）
# ============================================================

def stratified_split(
    articles: List[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    按类别分层切分，每个类别内部优先按时间排序。
    
    策略：
    1. 按类别分组
    2. 每组内：有日期的按日期升序排列，无日期的随机排在末尾
    3. 前 80% → train，中间 10% → val，后 10% → test
    4. 这样 test 集偏新（可检验泛化能力），train 集覆盖历史
    """
    rng = random.Random(seed)
    by_cat = collections.defaultdict(list)
    for a in articles:
        by_cat[a["category"]].append(a)

    train_all, val_all, test_all = [], [], []

    for cat in sorted(by_cat.keys()):
        group = by_cat[cat]
        # 分有日期和无日期
        with_date = [a for a in group if a["date"]]
        without_date = [a for a in group if not a["date"]]
        # 有日期的按时间排序
        with_date.sort(key=lambda a: a["date"])
        # 无日期的随机排序
        rng.shuffle(without_date)
        # 合并：有日期在前，无日期在后
        ordered = with_date + without_date

        n = len(ordered)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test

        train_all.extend(ordered[:n_train])
        val_all.extend(ordered[n_train:n_train + n_val])
        test_all.extend(ordered[n_train + n_val:])

    # 训练集内部打乱（确保不同类别混合）
    rng.shuffle(train_all)

    return train_all, val_all, test_all


# ============================================================
# 过采样（少数类别）
# ============================================================

def oversample_minority(
    articles: List[dict],
    min_count: int = 80,
    max_factor: float = 2.0,
    seed: int = 42,
) -> List[dict]:
    """
    对少于 min_count 篇的类别进行适度过采样（最多 max_factor 倍）。
    仅用于训练集。
    """
    rng = random.Random(seed)
    by_cat = collections.defaultdict(list)
    for a in articles:
        by_cat[a["category"]].append(a)

    result = []
    oversample_info = {}

    for cat in sorted(by_cat.keys()):
        group = by_cat[cat]
        original_count = len(group)
        result.extend(group)

        if original_count < min_count:
            # 计算需要补充多少（不超过 max_factor 倍）
            target = min(int(original_count * max_factor), min_count)
            extra_needed = target - original_count
            if extra_needed > 0:
                extras = rng.choices(group, k=extra_needed)
                # 标记为过采样样本
                for e in extras:
                    e = dict(e)
                    e["_oversampled"] = True
                    result.append(e)
                oversample_info[cat] = {
                    "original": original_count,
                    "added": extra_needed,
                    "total": original_count + extra_needed,
                }

    rng.shuffle(result)
    return result, oversample_info


# ============================================================
# 概要 & 意向提取（规则式）
# ============================================================

def extract_summary(article: dict) -> str:
    """从文章正文中自动提取概要（2-3 句话，80-150 字）。"""
    body = article["body"]
    title = article["title"]

    paragraphs = [p.strip() for p in re.split(r'\n\n+', body) if p.strip()]
    if not paragraphs:
        return f"一篇关于「{title}」的文章。"

    summary_parts = []

    # 取前几个段落的首句
    for para in paragraphs[:4]:
        sentence_match = re.match(r'(.+?[。？！…])', para)
        if sentence_match:
            sentence = sentence_match.group(1)
            if len(sentence) >= 10:
                summary_parts.append(sentence)
                if len("".join(summary_parts)) >= 80:
                    break

    if len(summary_parts) < 2 and paragraphs:
        first_para = paragraphs[0]
        if len(first_para) > 100:
            summary_parts = [first_para[:100] + "……"]
        else:
            summary_parts = [first_para]

    # 找转折句
    if len("".join(summary_parts)) < 60:
        for para in paragraphs[2:6]:
            turn_match = re.search(r'((?:然而|但是|不过|可是|问题在于|关键是).+?[。？！])', para)
            if turn_match:
                summary_parts.append(turn_match.group(1))
                break

    summary = "".join(summary_parts)
    if len(summary) > 150:
        cut_pos = summary.rfind("。", 0, 150)
        if cut_pos > 50:
            summary = summary[:cut_pos + 1]
        else:
            summary = summary[:150] + "……"

    return summary


def extract_intention(article: dict, rng: random.Random) -> str:
    """根据文章类别和标题，提取核心写作意向。"""
    cat_code = article["category"]
    title = article["title"]
    intentions = CATEGORY_INTENTIONS.get(cat_code, ["个人随笔"])
    selected = rng.sample(intentions, min(2, len(intentions)))

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
# 3 种混合 Prompt 模板
# ============================================================

def generate_prompt_template_A(article: dict, cat_label: str, rng: random.Random) -> str:
    """
    模板 A（占 50%）：完整版 — 类别 + 标题 + 概要 + 意向
    最丰富的条件信号，推理时最可控。
    """
    summary = extract_summary(article)
    intention = extract_intention(article, rng)

    openers = [
        f"请以和菜头的风格，写一篇{cat_label}类文章。",
        f"用和菜头「槽边往事」的笔触，写一篇{cat_label}方向的文章。",
        f"以和菜头的口吻，写以下{cat_label}类文章。",
    ]
    opener = rng.choice(openers)

    return (
        f"{opener}\n"
        f"类别：{cat_label}\n"
        f"标题：{article['title']}\n"
        f"概要：{summary}\n"
        f"核心意向：{intention}"
    )


def generate_prompt_template_B(article: dict, cat_label: str, rng: random.Random) -> str:
    """
    模板 B（占 30%）：中等版 — 类别 + 标题 + 简短指引
    减少概要依赖，鼓励模型自由发挥。
    """
    intention = extract_intention(article, rng)

    templates = [
        f"写一篇{cat_label}类的文章，题目是「{article['title']}」。\n写作方向：{intention}",
        f"以「{article['title']}」为题，写一篇{cat_label}风格的散文。\n核心意向：{intention}",
        f"从{cat_label}的角度，写一篇关于「{article['title']}」的文章。\n方向：{intention}",
    ]
    return rng.choice(templates)


def generate_prompt_template_C(article: dict, cat_label: str, rng: random.Random) -> str:
    """
    模板 C（占 20%）：极简版 — 仅类别 + 标题
    训练模型在最少提示下生成，增强鲁棒性。
    """
    templates = [
        f"[{cat_label}] 写一篇关于「{article['title']}」的文章",
        f"[{cat_label}] 以「{article['title']}」为题写一篇文章",
        f"[{cat_label}] 今天聊聊「{article['title']}」",
    ]
    return rng.choice(templates)


def generate_prompt_no_category(article: dict, rng: random.Random) -> str:
    """
    基线模板（无 category）：仅标题 + 可选概要
    用于 baseline 对照实验。
    """
    summary = extract_summary(article)
    templates = [
        f"请以和菜头的风格写一篇文章。\n标题：{article['title']}\n概要：{summary}",
        f"用和菜头的风格写一篇「{article['title']}」",
        f"以和菜头的口吻，写一篇关于「{article['title']}」的文章。\n概要：{summary}",
    ]
    return rng.choice(templates)


def generate_instruction(article: dict, rng: random.Random, use_category: bool = True) -> str:
    """
    按比例混合选择 prompt 模板。
    use_category=True  → 改进版（含 category），A:B:C = 50:30:20
    use_category=False → 基线版（无 category）
    """
    if not use_category:
        return generate_prompt_no_category(article, rng)

    cat_code = article["category"]
    _, cat_label = CLASSIFIED_FILES.get(cat_code, ("", "未知"))

    roll = rng.random()
    if roll < 0.5:
        return generate_prompt_template_A(article, cat_label, rng)
    elif roll < 0.8:
        return generate_prompt_template_B(article, cat_label, rng)
    else:
        return generate_prompt_template_C(article, cat_label, rng)


# ============================================================
# 转换为训练格式
# ============================================================

def messages_to_chatml(messages: list) -> str:
    """
    将 [{"role":...,"content":...},...] 转换为 ChatML 纯文本字符串。
    
    关键：TRL v0.24（Unsloth 锁定版本）要求 prompt/completion 是纯字符串，
    不能是 list[dict]。否则 Unsloth 会抛出：
      RuntimeError: You must specify a formatting_func
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def article_to_prompt_completion(article: dict, rng: random.Random,
                                  use_category: bool = True) -> dict:
    """
    将单篇文章转换为 prompt/completion 训练样本。
    
    输出格式：prompt 和 completion 均为 ChatML 纯文本字符串。
    TRL v0.24 的 SFTTrainer 会自动对 completion 部分计算 loss，
    prompt 部分不参与梯度计算 → completion-only loss。
    """
    instruction = generate_instruction(article, rng, use_category=use_category)

    # 构造 prompt 部分：system + user（ChatML 纯文本字符串）
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt_str = messages_to_chatml(prompt_messages)
    # 在 prompt 末尾加上 assistant 的起始标记，让 completion 紧接其后
    prompt_str += "\n<|im_start|>assistant\n"

    # 构造 completion 部分：纯文章正文（紧跟 prompt 的 assistant 标记）
    output_text = f"# {article['title']}\n\n{article['body']}"
    completion_str = output_text + "<|im_end|>"

    return {
        "prompt": prompt_str,
        "completion": completion_str,
        "_meta": {
            "title": article["title"],
            "category": article["category"],
            "date": article["date"],
            "char_count": article["char_count"],
        }
    }


def article_to_messages(article: dict, rng: random.Random,
                         use_category: bool = True) -> dict:
    """旧版：转换为 ChatML messages 格式（兼容旧训练脚本）。"""
    instruction = generate_instruction(article, rng, use_category=use_category)
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
# 统计 & 质量检查
# ============================================================

def compute_stats(samples: List[dict], data_format: str) -> dict:
    """计算数据集统计信息。"""
    token_counts = []
    char_counts = []
    cat_counts = collections.Counter()

    for s in samples:
        meta = s["_meta"]
        cat_counts[meta["category"]] += 1
        char_counts.append(meta["char_count"])

        if data_format == "messages":
            full_text = "".join(m["content"] for m in s["messages"])
        else:
            # v4.0: prompt 和 completion 是纯字符串
            full_text = s["prompt"] + s["completion"]
        token_counts.append(estimate_tokens(full_text))

    token_counts.sort()
    char_counts.sort()
    n = len(samples)

    def percentile(sorted_list, p):
        idx = min(int(len(sorted_list) * p / 100), len(sorted_list) - 1)
        return sorted_list[idx]

    return {
        "count": n,
        "tokens": {
            "total": sum(token_counts),
            "mean": sum(token_counts) // n if n else 0,
            "median": token_counts[n // 2] if n else 0,
            "p95": percentile(token_counts, 95) if n else 0,
            "max": token_counts[-1] if n else 0,
            "min": token_counts[0] if n else 0,
        },
        "chars": {
            "mean": sum(char_counts) // n if n else 0,
            "median": char_counts[n // 2] if n else 0,
            "p95": percentile(char_counts, 95) if n else 0,
            "max": char_counts[-1] if n else 0,
            "min": char_counts[0] if n else 0,
        },
        "cat_counts": dict(cat_counts),
        "token_distribution": {
            t: sum(1 for tc in token_counts if tc <= t)
            for t in [1024, 2048, 3072, 4096, 5120, 6144]
        },
    }


def print_stats(label: str, stats: dict):
    """打印统计信息。"""
    n = stats["count"]
    print(f"\n  [{label}] {n} 条样本")
    print(f"    Token: 均值={stats['tokens']['mean']}, 中位={stats['tokens']['median']}, "
          f"P95={stats['tokens']['p95']}, 最大={stats['tokens']['max']}")
    print(f"    字符: 均值={stats['chars']['mean']}, P95={stats['chars']['p95']}")
    print(f"    类别分布:")
    for cat in sorted(stats["cat_counts"].keys()):
        cnt = stats["cat_counts"][cat]
        _, label_name = CLASSIFIED_FILES.get(cat, ("", "?"))
        pct = cnt / n * 100 if n else 0
        print(f"      {cat} {label_name}: {cnt} ({pct:.1f}%)")
    print(f"    Token 覆盖:")
    for threshold, count in sorted(stats["token_distribution"].items()):
        pct = count / n * 100 if n else 0
        print(f"      ≤{threshold}: {count} ({pct:.1f}%)")


def show_random_samples(samples: List[dict], n: int = 5, seed: int = 42, data_format: str = "prompt"):
    """随机展示 n 条样本，标注类别。"""
    rng = random.Random(seed + 999)
    indices = rng.sample(range(len(samples)), min(n, len(samples)))
    print(f"\n  随机抽样 {len(indices)} 条样本展示:")

    for idx in indices:
        s = samples[idx]
        meta = s["_meta"]
        cat_code = meta["category"]
        _, cat_label = CLASSIFIED_FILES.get(cat_code, ("", "?"))

        print(f"\n  {'─' * 50}")
        print(f"  #{idx} | 类别: {cat_code}({cat_label}) | 标题: {meta['title']} | "
              f"字数: {meta['char_count']} | 日期: {meta.get('date', 'N/A')}")

        if data_format == "messages":
            user_msg = s["messages"][1]["content"]
            asst_preview = s["messages"][2]["content"][:120]
        else:
            # v4.0: prompt/completion 是纯字符串
            user_msg = s["prompt"][:200]
            asst_preview = s["completion"][:120]

        print(f"  [User] {user_msg[:200]}")
        print(f"  [Asst] {asst_preview}...")


# ============================================================
# 数据泄漏检查
# ============================================================

def check_data_leakage(train: List[dict], val: List[dict], test: List[dict]):
    """检查 train/val/test 之间是否有标题或正文重叠。"""
    def get_titles(samples):
        return {s["_meta"]["title"] for s in samples}

    train_titles = get_titles(train)
    val_titles = get_titles(val)
    test_titles = get_titles(test)

    leak_tv = train_titles & val_titles
    leak_tt = train_titles & test_titles
    leak_vt = val_titles & test_titles

    ok = True
    if leak_tv:
        print(f"  [警告] train/val 标题重叠 {len(leak_tv)} 篇: {list(leak_tv)[:3]}")
        ok = False
    if leak_tt:
        print(f"  [警告] train/test 标题重叠 {len(leak_tt)} 篇: {list(leak_tt)[:3]}")
        ok = False
    if leak_vt:
        print(f"  [警告] val/test 标题重叠 {len(leak_vt)} 篇: {list(leak_vt)[:3]}")
        ok = False

    if ok:
        print("  [通过] 无数据泄漏：train/val/test 之间没有标题重叠")

    return ok


# ============================================================
# 文件写入
# ============================================================

def write_jsonl(filepath: Path, samples: List[dict]):
    """写入 JSONL，去掉 _meta 字段。"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in samples:
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="将和菜头语料转换为 LoRA 微调训练数据 v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default="training_data",
                        help="输出目录（默认 training_data/）")
    parser.add_argument("--preview", type=int, default=0,
                        help="只预览前 N 条，不写文件")
    parser.add_argument("--max-length", type=int, default=6000,
                        help="单篇最大字符数（默认 6000）")
    parser.add_argument("--min-length", type=int, default=200,
                        help="单篇最小字符数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--format", choices=["enhanced", "messages"],
                        default="enhanced",
                        help="enhanced=prompt/completion（推荐），messages=旧版 ChatML")
    parser.add_argument("--oversample", action="store_true",
                        help="对少数类别（C/D）在训练集中进行 2x 过采样")
    parser.add_argument("--baseline", action="store_true",
                        help="生成无 category 的基线数据集（用于对照实验）")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="训练集比例（默认 0.8）")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例（默认 0.1）")
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0.01:
        print("[错误] train_ratio + val_ratio 不能超过 0.99")
        sys.exit(1)

    rng = random.Random(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("和菜头语料 → LoRA 训练数据 转换工具 v4.0")
    print("=" * 60)
    print(f"  格式: {args.format}")
    print(f"  类别信号: {'无（baseline）' if args.baseline else '有（改进版）'}")
    print(f"  过采样: {'是' if args.oversample else '否'}")
    print(f"  切分: train {args.train_ratio:.0%} / val {args.val_ratio:.0%} / test {test_ratio:.0%}")

    # ---- Step 1: 加载文章 ----
    print(f"\n[Step 1] 加载分类文章...")
    articles = load_all_articles()
    print(f"  共加载 {len(articles)} 篇文章")

    # ---- Step 2: 去重 ----
    print(f"\n[Step 2] 去重检查...")
    articles, dedup_stats = deduplicate(articles)
    n_dup_title = len(dedup_stats["dup_title"])
    n_dup_hash = len(dedup_stats["dup_hash"])
    print(f"  去重后: {len(articles)} 篇（去除标题重复 {n_dup_title}，正文重复 {n_dup_hash}）")
    if n_dup_title > 0:
        for line in dedup_stats["dup_title"][:5]:
            print(line)
    if n_dup_hash > 0:
        for line in dedup_stats["dup_hash"][:5]:
            print(line)

    # ---- Step 3: 过滤长度 ----
    print(f"\n[Step 3] 过滤（{args.min_length} ≤ 字数 ≤ {args.max_length}）...")
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
    print(f"  保留 {len(filtered)} 篇 | 跳过过短 {skipped_short} | 截断过长 {truncated}")

    # ---- 类别分布 ----
    cat_dist = collections.Counter(a["category"] for a in filtered)
    print(f"\n  类别分布（过滤后）:")
    for cat in sorted(cat_dist.keys()):
        _, label = CLASSIFIED_FILES.get(cat, ("", "?"))
        print(f"    {cat} {label}: {cat_dist[cat]} 篇 ({cat_dist[cat]/len(filtered)*100:.1f}%)")

    # ---- Step 4: 分层切分 ----
    print(f"\n[Step 4] 分层切分（文章级，时间优先）...")
    train_articles, val_articles, test_articles = stratified_split(
        filtered,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
    )

    # 过采样（仅训练集）
    oversample_info = {}
    if args.oversample:
        print(f"\n[Step 4.5] 少数类别过采样...")
        train_articles, oversample_info = oversample_minority(
            train_articles, min_count=80, max_factor=2.0, seed=args.seed
        )
        if oversample_info:
            for cat, info in oversample_info.items():
                _, label = CLASSIFIED_FILES.get(cat, ("", "?"))
                print(f"  {cat} {label}: {info['original']} → {info['total']}（+{info['added']}）")
        else:
            print(f"  无需过采样（所有类别 ≥ 80 篇）")

    print(f"\n  切分结果:")
    print(f"    训练集: {len(train_articles)} 篇")
    print(f"    验证集: {len(val_articles)} 篇")
    print(f"    测试集: {len(test_articles)} 篇")

    # 打印每个 split 的类别分布
    for name, split_articles in [("train", train_articles), ("val", val_articles), ("test", test_articles)]:
        cat_c = collections.Counter(a["category"] for a in split_articles)
        parts = [f"{c}:{cat_c.get(c,0)}" for c in sorted(CLASSIFIED_FILES.keys())]
        print(f"    {name}: {' | '.join(parts)}")

    # ---- Step 5: 转换为训练格式 ----
    print(f"\n[Step 5] 生成训练样本（模板 A:B:C = 50:30:20）...")

    use_category = not args.baseline

    def convert_articles(article_list):
        if args.format == "messages":
            return [article_to_messages(a, rng, use_category=use_category) for a in article_list]
        else:
            return [article_to_prompt_completion(a, rng, use_category=use_category) for a in article_list]

    train_samples = convert_articles(train_articles)
    val_samples = convert_articles(val_articles)
    test_samples = convert_articles(test_articles)

    data_key = "messages" if args.format == "messages" else "prompt"
    data_format_str = "messages" if args.format == "messages" else "prompt/completion"

    # ---- Step 6: 统计 ----
    print(f"\n[Step 6] 数据集统计")

    train_stats = compute_stats(train_samples, args.format)
    val_stats = compute_stats(val_samples, args.format)
    test_stats = compute_stats(test_samples, args.format)

    print_stats("训练集", train_stats)
    print_stats("验证集", val_stats)
    print_stats("测试集", test_stats)

    # ---- 数据泄漏检查 ----
    print(f"\n[Step 7] 数据泄漏检查...")
    check_data_leakage(train_samples, val_samples, test_samples)

    # ---- 随机展示样本 ----
    show_random_samples(train_samples, n=5, seed=args.seed, data_format=args.format)

    # ---- Completion-only loss 确认 ----
    print(f"\n[Loss 模式确认]")
    if args.format == "enhanced":
        print(f"  格式: prompt/completion → TRL 原生 completion-only loss")
        print(f"  仅在 assistant 回复（文章正文）部分计算 loss")
        print(f"  system 提示 + user 指令 不参与梯度计算")
    else:
        print(f"  格式: messages → 需在训练脚本中配置 completion-only loss")

    # ---- 预览模式 ----
    if args.preview > 0:
        print(f"\n[预览模式] 显示前 {args.preview} 条：")
        for i, s in enumerate(train_samples[:args.preview]):
            meta = s["_meta"]
            print(f"\n{'=' * 50}")
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
                print(f"  {s['prompt'][:400]}")
                print("[COMPLETION]")
                print(f"  {s['completion'][:200]}...")
        return

    # ---- Step 8: 写入文件 ----
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"
    test_file = output_dir / "test.jsonl"
    # 兼容旧脚本：同时写一份 eval.jsonl（= val.jsonl）
    eval_file = output_dir / "eval.jsonl"

    write_jsonl(train_file, train_samples)
    write_jsonl(val_file, val_samples)
    write_jsonl(test_file, test_samples)
    write_jsonl(eval_file, val_samples)  # 向后兼容

    print(f"\n[Step 8] 文件已写入:")
    for f in [train_file, val_file, test_file, eval_file]:
        size = os.path.getsize(f)
        unit = "KB" if size < 1024 * 1024 else "MB"
        val = size / 1024 if unit == "KB" else size / 1024 / 1024
        print(f"  {f}  ({val:.1f} {unit})")

    # ---- Step 9: 写入详细报告 ----
    report_file = output_dir / "data_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("和菜头 LoRA 微调数据集报告 v4.0\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"数据格式: {data_format_str}\n")
        f.write(f"类别信号: {'无（baseline）' if args.baseline else '有（改进版）'}\n")
        f.write(f"过采样: {'是' if args.oversample else '否'}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"字符限制: {args.min_length}-{args.max_length}\n\n")

        f.write("切分统计:\n")
        for name, stats in [("训练集", train_stats), ("验证集", val_stats), ("测试集", test_stats)]:
            f.write(f"\n  {name}: {stats['count']} 条\n")
            f.write(f"    Token: 均值={stats['tokens']['mean']}, 中位={stats['tokens']['median']}, "
                    f"P95={stats['tokens']['p95']}, 最大={stats['tokens']['max']}, 最小={stats['tokens']['min']}\n")
            f.write(f"    字符: 均值={stats['chars']['mean']}, P95={stats['chars']['p95']}\n")
            f.write(f"    类别:\n")
            for cat in sorted(CLASSIFIED_FILES.keys()):
                cnt = stats["cat_counts"].get(cat, 0)
                _, label = CLASSIFIED_FILES.get(cat, ("", "?"))
                pct = cnt / stats["count"] * 100 if stats["count"] else 0
                f.write(f"      {cat} {label}: {cnt} ({pct:.1f}%)\n")
            f.write(f"    Token 覆盖:\n")
            for threshold, count in sorted(stats["token_distribution"].items()):
                pct = count / stats["count"] * 100 if stats["count"] else 0
                f.write(f"      ≤{threshold}: {count} ({pct:.1f}%)\n")

        if oversample_info:
            f.write(f"\n过采样详情:\n")
            for cat, info in oversample_info.items():
                _, label = CLASSIFIED_FILES.get(cat, ("", "?"))
                f.write(f"  {cat} {label}: {info['original']} → {info['total']}（+{info['added']}）\n")

        f.write(f"\n去重报告:\n")
        f.write(f"  标题重复: {n_dup_title} 篇\n")
        f.write(f"  正文重复: {n_dup_hash} 篇\n")
        for line in dedup_stats["dup_title"]:
            f.write(f"  {line}\n")
        for line in dedup_stats["dup_hash"]:
            f.write(f"  {line}\n")

        f.write(f"\nPrompt 模板混合比例:\n")
        f.write(f"  模板 A（完整: 类别+标题+概要+意向）: 50%\n")
        f.write(f"  模板 B（中等: 类别+标题+意向）: 30%\n")
        f.write(f"  模板 C（极简: 类别+标题）: 20%\n")

        f.write(f"\nLoss 模式: completion-only（仅在文章正文部分计算 loss）\n")
        f.write(f"System Prompt: {SYSTEM_PROMPT}\n")

        f.write(f"\n过拟合判据建议:\n")
        f.write(f"  1. train_loss < 0.3 且 val_loss > 1.5 → 严重过拟合\n")
        f.write(f"  2. val_loss 连续 3 个 eval 不降 → 应停止训练\n")
        f.write(f"  3. 生成文本出现大段原文复读 → 过拟合\n")
        f.write(f"  4. 生成结果无论输入什么标题都很相似 → 模式坍缩\n")

        f.write(f"\n评价提示词建议:\n")
        f.write(f"  - 给定训练集外的标题，看生成文章是否符合和菜头风格\n")
        f.write(f"  - 对比 6 个类别的生成质量，弱类别是否可接受\n")
        f.write(f"  - 检查是否出现「死记硬背」（输出训练集原文片段）\n")

    print(f"  {report_file}")

    # ---- 完成提示 ----
    print(f"\n{'=' * 60}")
    print("数据准备完成！")
    print(f"\n数据格式: {data_format_str}")
    print(f"训练特性: completion-only loss（只学写作，不学生成指令）")
    if args.baseline:
        print(f"实验类型: 基线（无 category）")
    else:
        print(f"实验类型: 改进版（含 category）")
    print(f"\n下一步：运行训练脚本")
    print(f"  python train_hecaitou.py --data-dir {args.output}")
    print(f"\n对照实验（可选）：")
    print(f"  # 1) 生成基线数据")
    print(f"  python prepare_training_data.py --baseline --output training_data_baseline")
    print(f"  # 2) 生成改进数据")
    print(f"  python prepare_training_data.py --output training_data_improved")
    print(f"  # 3) 分别训练并对比")
    print(f"  python train_hecaitou.py --data-dir training_data_baseline --output-dir output_baseline")
    print(f"  python train_hecaitou.py --data-dir training_data_improved --output-dir output_improved")
    print("=" * 60)


if __name__ == "__main__":
    main()
