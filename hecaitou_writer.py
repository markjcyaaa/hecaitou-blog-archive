#!/usr/bin/env python3
"""
和菜头风格自动写作工作流 v4.0
================================
基于本地 Ollama + Qwen 模型，自动完成：
  事件分类（多类型叠加）→ 检索参考原文 → 三稿并行生成 → 自检(50%通过)
  → 去AI化审查 → 三稿比较 → 缺陷迭代 → 输出最佳文章

四种使用方式：
  1. 命令行单次生成：
     python hecaitou_writer.py --topic "最近大家都在用AI写东西" --words 2000
     python hecaitou_writer.py --topic "某知名程序员猝死" --words 1500 --type C

  2. 交互模式（推荐，适合反复调整）：
     python hecaitou_writer.py --interactive

  3. 作为模块调用：
     from hecaitou_writer import run_workflow, ArticleRequest
     result = run_workflow(ArticleRequest(topic="xxx"))

  4. 快速模式（跳过三稿，只生成一篇）：
     python hecaitou_writer.py --topic "xxx" --quick

依赖：
  pip install requests
  本地需运行 ollama serve，并已拉取模型（如 qwen3.5:35b-a3b 或 qwen3:30b-a3b）
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# ============================================================
# 全局配置（可通过 CLI 参数或 configure() 覆盖）
# ============================================================

_config = {
    "ollama_url": "http://localhost:11434",
    "writer_model": "qwen3.5:35b-a3b",
    "critic_model": "qwen3.5:35b-a3b",
    "max_iterations": 3,
    "score_threshold": 80,
    "default_word_count": 2000,
    # 上下文窗口：qwen3.5:35b-a3b 支持最大 131072，但实际受限于 RAM
    # 32GB RAM 建议设 16384~32768；16GB RAM 建议 8192
    "num_ctx": 16384,
    # 参考原文策略："full"=全文, "smart"=根据上下文窗口自动裁剪
    "reference_mode": "smart",
    # 单篇参考原文最大字符数（仅 smart 模式生效）
    "max_ref_chars": 3000,
}

# Skill 文件和原文的位置
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_FILE = SCRIPT_DIR / "HECAITOU_STYLE_SKILLS.md"
PASSAGES_DIR = SCRIPT_DIR / "passages"
CLASSIFIED_DIR = SCRIPT_DIR / "classified"

# 分类文件映射（类型代码 → 文件名）
CLASSIFIED_FILES = {
    "A": "A_社会观察.md",
    "B": "B_技术产品.md",
    "C": "C_生死无常.md",
    "D": "D_自省修行.md",
    "E": "E_文化阅读.md",
    "F": "F_日常生活.md",
}


def configure(**kwargs):
    """更新全局配置。示例: configure(writer_model="qwen3:30b-a3b", num_ctx=8192)"""
    for k, v in kwargs.items():
        if k in _config:
            _config[k] = v
        else:
            print(f"[警告] 未知配置项: {k}")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ArticleRequest:
    topic: str
    word_count: int = 2000
    article_type: Optional[str] = None  # A/B/C/D/E/F，None=自动判断
    secondary_type: Optional[str] = None  # 副类型，None=无
    style_hint: Optional[str] = None    # "更毒舌" "更佛系" 等
    reference_article: Optional[str] = None  # 指定锚点原文标题
    quick_mode: bool = False  # 快速模式：跳过三稿，只生成一篇


@dataclass
class CriticResult:
    score: int
    style_score: int
    content_score: int
    structure_score: int
    problems: List[str]
    suggestions: str
    passed: bool


@dataclass
class WorkflowResult:
    status: str           # "success" | "max_iterations" | "error"
    article: str
    title: str
    article_type: str
    secondary_type: Optional[str] = None
    iterations: int = 0
    total_drafts: int = 0
    final_score: int = 0
    reference_articles: List[str] = field(default_factory=list)
    critic_history: List[CriticResult] = field(default_factory=list)
    compare_history: List[dict] = field(default_factory=list)


# ============================================================
# Ollama API 调用
# ============================================================

def call_ollama(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    think: bool = False,
    temperature: float = 0.7,
    timeout: int = 600,
    num_ctx: Optional[int] = None,
) -> str:
    """调用本地 Ollama API，返回生成的文本。"""
    model = model or _config["writer_model"]
    num_ctx = num_ctx or _config["num_ctx"]
    url = f"{_config['ollama_url']}/api/chat"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")

        # 打印 token 统计（如果有）
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        total_duration = data.get("total_duration", 0)
        if eval_count and total_duration:
            speed = eval_count / (total_duration / 1e9) if total_duration else 0
            print(f"  [token] 输入={prompt_eval_count} 输出={eval_count} "
                  f"速度={speed:.1f} tok/s")

        return content

    except requests.exceptions.ConnectionError:
        print("\n[错误] 无法连接到 Ollama。请确认：")
        print("  1. Ollama 已启动 (ollama serve)")
        print(f"  2. 模型已下载 (ollama pull {model})")
        print(f"  3. 地址正确 ({_config['ollama_url']})")
        return ""
    except requests.exceptions.Timeout:
        print(f"\n[错误] Ollama 响应超时 ({timeout}s)")
        print("  可能原因：模型太慢、内存不足、或 num_ctx 太大")
        print(f"  当前 num_ctx={num_ctx}，尝试减小：configure(num_ctx=8192)")
        return ""
    except Exception as e:
        print(f"\n[错误] Ollama 调用失败: {e}")
        return ""


def check_ollama_ready() -> bool:
    """检查 Ollama 服务和模型是否就绪，自动切换可用模型。"""
    try:
        resp = requests.get(f"{_config['ollama_url']}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]

        writer_base = _config["writer_model"].split(":")[0]
        if not any(writer_base in m for m in models):
            print(f"[警告] 未找到模型 {_config['writer_model']}")
            print(f"  可用模型：{models}")
            if models:
                # 优先选择 qwen 系列
                qwen_models = [m for m in models if "qwen" in m.lower()]
                fallback = qwen_models[0] if qwen_models else models[0]
                print(f"  [自动切换] → {fallback}")
                _config["writer_model"] = fallback
                _config["critic_model"] = fallback
            else:
                print("[错误] 没有可用模型。请先运行 ollama pull <model>")
                return False
        return True

    except requests.exceptions.ConnectionError:
        print("[错误] 无法连接到 Ollama。请确认 ollama serve 已启动。")
        return False
    except Exception as e:
        print(f"[错误] 检查 Ollama 失败: {e}")
        return False


# ============================================================
# 原文加载与检索
# ============================================================

_article_cache: Optional[List[dict]] = None


def load_all_articles(force_reload: bool = False) -> List[dict]:
    """
    加载所有原文，解析为 [{title, content, file, date, category, subcategory}] 列表。
    优先从 classified/ 目录加载（按类型整理好的文件）。
    如果 classified/ 不存在，回退到 passages/ 目录。
    结果会缓存，避免重复解析。
    """
    global _article_cache
    if _article_cache is not None and not force_reload:
        return _article_cache

    articles = []

    # 优先从 classified/ 目录加载
    classified_files = []
    for cat_code, fname in CLASSIFIED_FILES.items():
        fpath = CLASSIFIED_DIR / fname
        if fpath.exists():
            classified_files.append((cat_code, fpath))
        # 也支持与脚本同目录
        alt_path = SCRIPT_DIR / fname
        if alt_path.exists() and alt_path != fpath:
            classified_files.append((cat_code, alt_path))

    if classified_files:
        # 从分类文件加载
        for cat_code, fpath in classified_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            current_subcat = ""
            # 按 ### 分割出各篇文章
            parts = re.split(r'\n### ', content)
            for part in parts[1:]:  # 跳过文件头
                lines = part.strip().split("\n")
                title = lines[0].strip()
                if not title or title.startswith("目录"):
                    continue

                body = "\n".join(lines[1:]).strip()
                if len(body) < 50:
                    continue

                # 提取日期
                date_match = re.search(r'\*\*日期\*\*[：:]\s*(\d{4}-\d{2}-\d{2})', body)
                date = date_match.group(1) if date_match else ""

                # 提取子分类（从上面最近的 ## 标题获取）
                # 回溯找到当前所在的 ## 子分类
                subcat_match = re.search(
                    r'## ((?:A|B|C|D|E|F)\d_\S+)',
                    content[:content.index(f'### {title}')][::-1][:500][::-1]
                )

                articles.append({
                    "title": title,
                    "content": body,
                    "file": fname,
                    "date": date,
                    "category": cat_code,
                })

        print(f"[原文库] 从分类文件加载 {len(articles)} 篇文章，"
              f"来自 {len(classified_files)} 个文件")
    else:
        # 回退到 passages/ 目录
        md_patterns = [
            str(PASSAGES_DIR / "*.md"),
            str(SCRIPT_DIR / "20??-??.md"),
        ]
        md_files = []
        for pattern in md_patterns:
            md_files.extend(glob.glob(pattern))
        md_files = sorted(set(md_files))

        if not md_files:
            print(f"[警告] 未找到原文 md 文件")
            print(f"  搜索路径：{CLASSIFIED_DIR}, {PASSAGES_DIR}, {SCRIPT_DIR}")
            return articles

        for fpath in md_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            parts = re.split(r'\n## ', content)
            for part in parts[1:]:
                lines = part.strip().split("\n")
                title = lines[0].strip()
                if title == "目录":
                    continue
                body = "\n".join(lines[1:]).strip()
                if len(body) < 50:
                    continue

                date_match = re.search(r'\*\*日期\*\*:\s*(\d{4}-\d{2}-\d{2})', body)
                date = date_match.group(1) if date_match else fname.replace(".md", "")

                articles.append({
                    "title": title,
                    "content": body,
                    "file": fname,
                    "date": date,
                    "category": "",
                })

        print(f"[原文库] 从月度文件加载 {len(articles)} 篇文章，"
              f"来自 {len(md_files)} 个文件")

    _article_cache = articles
    return articles


def search_articles(
    articles: List[dict],
    query: str,
    top_k: int = 3,
    article_type: Optional[str] = None,
) -> List[dict]:
    """
    基于关键词的原文检索。
    当文章有 category 字段时（从分类文件加载），优先返回同类型文章。
    """
    if not articles:
        return []

    recommended_titles = set()
    if article_type:
        recommended_titles = _get_recommended_titles(article_type)

    # 提取查询关键词
    stopwords = set("的了是在我你他她它这那有也和与或但如果因为所以因此"
                    "可以能够已经正在一个不会就是都还要到被把让给从对于而且")
    query_words = set(re.findall(r'[\u4e00-\u9fff]{2,}', query))
    query_words |= set(re.findall(r'[a-zA-Z]{2,}', query.lower()))
    query_words -= stopwords

    scored = []
    for art in articles:
        score = 0.0
        text = art["title"] + " " + art["content"][:1000]

        # 关键词匹配
        text_words = set(re.findall(r'[\u4e00-\u9fff]{2,}', text))
        text_words |= set(re.findall(r'[a-zA-Z]{2,}', text.lower()))
        overlap = len(query_words & text_words)
        score += overlap

        # 标题匹配加权 (3x)
        title_words = set(re.findall(r'[\u4e00-\u9fff]{2,}', art["title"]))
        title_overlap = len(query_words & title_words)
        score += title_overlap * 3

        # Skill 推荐原文加权 (5x)
        if art["title"] in recommended_titles:
            score += 5

        # 同类型加权 (3x) — 利用分类文件的 category 字段
        if article_type and art.get("category") == article_type:
            score += 3

        scored.append((score, art))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored[:top_k] if item[0] > 0]

    # 补充不足
    if len(results) < 2:
        import random
        if recommended_titles:
            for art in articles:
                if art["title"] in recommended_titles and art not in results:
                    results.append(art)
                    if len(results) >= top_k:
                        break
        # 优先从同类型补充
        if len(results) < 2 and article_type:
            same_type = [a for a in articles
                         if a.get("category") == article_type and a not in results]
            if same_type:
                results.extend(random.sample(same_type,
                                             min(top_k - len(results), len(same_type))))
        if len(results) < 2:
            remaining = [a for a in articles if a not in results]
            if remaining:
                results.extend(random.sample(remaining, min(2, len(remaining))))

    return results[:top_k]


def find_article_by_title(articles: List[dict], title: str) -> Optional[dict]:
    """按标题精确/模糊查找单篇文章。"""
    # 精确匹配
    for art in articles:
        if art["title"] == title:
            return art
    # 模糊匹配
    for art in articles:
        if title in art["title"] or art["title"] in title:
            return art
    return None


def _get_recommended_titles(article_type: str) -> set:
    """从 Skill 文件的参考索引中提取对应类型的推荐原文标题。"""
    # 硬编码关键推荐原文（与 Skill 文件第七章保持同步）
    recommendations = {
        "A": {"甘心和自己无关", "松鼠尾巴", "帮我把照片里的其他人P掉",
               "再也不可能遇见那么好的人了", "恨的教育"},
        "B": {"来自苹果的重击", "Google 做了淘宝的事", "艺术家未遂第三年",
               "库克的防腐工程", "一次成功的个人反制", "翻译：创始人模式"},
        "C": {"仅有他汀是不够的", "罗振宇聋了之后", "一年之得", "中年男子的最大福报"},
        "D": {"不善的缘起", "六个比喻", "修行的姿势", "盲盒、黑盒以及错误", "有所不为"},
        "E": {"年度读书总结（2025）", "历史的细节", "职业罪犯总是会清理现场",
               "成功学与三个保罗"},
        "F": {"写作者快乐的一天", "午睡悲伤综合征", "猫过马路", "挥别小笼包"},
    }
    return recommendations.get(article_type.upper(), set())


# ============================================================
# Skill 文件加载
# ============================================================

_skill_cache: Optional[str] = None


def load_skill() -> str:
    """加载 Skill 文件内容（带缓存）。"""
    global _skill_cache
    if _skill_cache is not None:
        return _skill_cache

    if SKILL_FILE.exists():
        with open(SKILL_FILE, "r", encoding="utf-8") as f:
            _skill_cache = f.read()
        return _skill_cache
    else:
        print(f"[警告] 未找到 Skill 文件：{SKILL_FILE}")
        _skill_cache = ""
        return ""


# ============================================================
# Step 1：事件分类
# ============================================================

ARTICLE_TYPES = {
    "A": "社会观察型",
    "B": "技术/产品体验型",
    "C": "生死/无常型",
    "D": "自省/修行型",
    "E": "文化/阅读型",
    "F": "日常生活型",
}


def classify_event(topic: str) -> Tuple[str, Optional[str]]:
    """让模型判断事件属于哪种文章类型，返回 (主类型, 副类型或None)。
    允许多类型叠加。"""
    prompt = f"""请判断以下事件/话题最适合用哪种文章类型来写：

事件：{topic}

文章类型选项：
A - 社会观察型（社会新闻、网络争议、公众人物言行、群体行为）
B - 技术/产品体验型（新产品发布、AI/科技更新、消费陷阱）
C - 生死/无常型（人物逝世、重大灾难、健康危机）
D - 自省/修行型（个人反思、习惯改变、价值观整理）
E - 文化/阅读型（书评、电影、历史、文化现象）
F - 日常生活型（日常小事、季节、饮食、宠物）

如果事件明确属于一种类型，只回答一个字母。
如果事件横跨两种类型，回答两个字母（主类型在前，副类型在后），用空格分隔。
例如："C B" 表示主类型是生死/无常型，副类型是技术/产品体验型。
只回答字母，不要解释。"""

    result = call_ollama(prompt, model=_config["writer_model"],
                         think=False, temperature=0.1, num_ctx=2048)
    if not result:
        return ("A", None)

    # 提取所有出现的类型字母
    found_types = []
    for char in result.upper():
        if char in ARTICLE_TYPES and char not in found_types:
            found_types.append(char)

    if not found_types:
        return ("A", None)
    elif len(found_types) == 1:
        return (found_types[0], None)
    else:
        return (found_types[0], found_types[1])


# ============================================================
# Step 2 & 3：生成文章
# ============================================================

def _prepare_reference_texts(
    references: List[dict],
    skill_len: int,
    word_count: int,
) -> List[Tuple[str, str]]:
    """
    根据参考策略和上下文窗口，准备参考原文。
    返回 [(title, text), ...]。

    策略：
    - "full"：使用完整原文
    - "smart"：根据剩余 context 空间自动分配
    """
    mode = _config["reference_mode"]
    num_ctx = _config["num_ctx"]

    if mode == "full":
        return [(r["title"], r["content"]) for r in references]

    # smart 模式：计算可用空间
    # Qwen tokenizer: 中文约 0.7 token/字, 英文约 0.3 token/word
    # 保守按 0.8 token/字 估算
    CHARS_TO_TOKENS = 0.8
    total_tokens = num_ctx
    # 预留空间：system prompt (skill) + 生成空间 + 指令 + 格式
    skill_tokens = int(skill_len * CHARS_TO_TOKENS)
    output_tokens = int(word_count * CHARS_TO_TOKENS)
    overhead_tokens = 800  # 指令、格式、分类结果等
    available_tokens = total_tokens - skill_tokens - output_tokens - overhead_tokens

    if available_tokens <= 500:
        # 上下文太小，只用标题 + 简短摘要
        print(f"  [注意] 上下文窗口 ({num_ctx}) 较紧张，参考原文将被大幅压缩")
        print(f"  提示：使用 --ctx 32768 或 /ctx 32768 增大上下文")
        return [(r["title"], r["content"][:200] + "...")
                for r in references]

    available_chars = int(available_tokens / CHARS_TO_TOKENS)
    per_ref_chars = available_chars // max(len(references), 1)

    # 确保至少 500 字/篇
    per_ref_chars = max(per_ref_chars, 500)
    # 不超过配置上限
    max_chars = _config["max_ref_chars"]
    per_ref_chars = min(per_ref_chars, max_chars)

    result = []
    for r in references:
        content = r["content"]
        if len(content) <= per_ref_chars:
            # 全文
            result.append((r["title"], content))
        else:
            # 智能裁剪：保留开头 + 结尾
            head_len = int(per_ref_chars * 0.6)
            tail_len = int(per_ref_chars * 0.3)
            head = content[:head_len]
            tail = content[-tail_len:]
            trimmed = f"{head}\n\n[...原文中间部分省略...]\n\n{tail}"
            result.append((r["title"], trimmed))

    return result


def build_writer_prompt(
    topic: str,
    article_type: str,
    word_count: int,
    reference_texts: List[dict],
    previous_critique: Optional[str] = None,
    style_hint: Optional[str] = None,
    secondary_type: Optional[str] = None,
    draft_variation: Optional[str] = None,
    avoid_issues: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    构建写作 prompt。
    返回 (system_prompt, user_prompt)。
    draft_variation: 草稿差异指令（如"从不同的切入角度"）
    avoid_issues: 本轮需规避的问题清单
    """
    skill_text = load_skill()

    # System Prompt = Skill 文件全文
    system = skill_text if skill_text else (
        "你是和菜头，中文互联网上活跃超过25年的写作者。"
        "你运营公众号「槽边往事」，语气温和但刻薄，善用自嘲和冷幽默，"
        "短句为主，口语化书面语，第一人称叙述。"
    )

    # 准备参考原文（根据上下文策略）
    prepared_refs = _prepare_reference_texts(
        reference_texts, len(skill_text), word_count
    )

    type_name = ARTICLE_TYPES.get(article_type, "社会观察型")

    user_parts = []
    user_parts.append("# 写作任务\n")
    user_parts.append(f"**事件/话题**：{topic}")
    user_parts.append(f"**主类型**：{article_type} - {type_name}")
    if secondary_type:
        sec_name = ARTICLE_TYPES.get(secondary_type, "")
        user_parts.append(f"**副类型**：{secondary_type} - {sec_name}")
        user_parts.append(f"**融合策略**：主框架用{type_name}的结构模板，在适当段落融入{sec_name}的元素")
    user_parts.append(f"**字数要求**：约 {word_count} 字")

    if style_hint:
        user_parts.append(f"**风格微调**：{style_hint}")

    if draft_variation:
        user_parts.append(f"\n**本篇差异要求**：{draft_variation}")

    if avoid_issues:
        user_parts.append("\n**本轮需规避的问题**（上一轮发现的缺陷，务必避免）：")
        for i, issue in enumerate(avoid_issues, 1):
            user_parts.append(f"  {i}. {issue}")

    # 附上参考原文（完整或智能裁剪）
    if prepared_refs:
        user_parts.append("\n# 参考原文（请仔细阅读，模仿这些文章的语感、节奏和结构）\n")
        for i, (title, text) in enumerate(prepared_refs, 1):
            char_count = len(text)
            user_parts.append(
                f"### 参考 {i}：《{title}》（{char_count}字）\n\n{text}\n"
            )

    # 迭代时附上批评
    if previous_critique:
        user_parts.append(
            f"\n# 上一轮评审意见（请针对这些问题改进）\n{previous_critique}\n"
        )

    user_parts.append(
        "\n请直接输出文章。格式：第一行写标题，空一行后写正文。"
        "不要输出任何说明、注释或 markdown 标记。"
    )

    return system, "\n".join(user_parts)


def generate_article(
    topic: str,
    article_type: str,
    word_count: int,
    reference_texts: List[dict],
    previous_critique: Optional[str] = None,
    style_hint: Optional[str] = None,
    secondary_type: Optional[str] = None,
    draft_variation: Optional[str] = None,
    avoid_issues: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    调用模型生成文章。
    返回 (title, content)。
    """
    system, user = build_writer_prompt(
        topic, article_type, word_count, reference_texts, previous_critique,
        style_hint, secondary_type, draft_variation, avoid_issues,
    )

    raw = call_ollama(
        user, system=system,
        model=_config["writer_model"],
        think=False,
        temperature=0.75,
    )

    if not raw:
        return topic[:20], "[生成失败，请检查模型状态]"

    # 解析标题和正文
    lines = raw.strip().split("\n")
    title = ""
    content_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped
            content_start = i + 1
            break

    # 跳过标题后的空行
    while content_start < len(lines) and not lines[content_start].strip():
        content_start += 1

    content = "\n".join(lines[content_start:]).strip()
    if not title:
        title = topic[:20]
    if not content:
        content = raw

    return title, content


# ============================================================
# Step 4：评分
# ============================================================

CRITIC_SYSTEM = """你是一个严格的文学评论家，专门评判"和菜头"风格的文章。

评分标准（每项0-100分）：
1. 风格相似度：是否有和菜头的语感？短句节奏、自嘲、冷幽默、"温和的刻薄"、设问自答？
2. 内容质量：切入点是否具体？有没有生动比喻？论证是否有层次？结尾是否温暖或平和？
3. 结构完整度：是否遵循了对应文章类型的结构模板？

扣分项检查：
- 使用了网络流行语？（严重扣分 -20）
- 使用了鸡汤句式或"值得深思"之类的假大空？（严重扣分 -20）
- 像政策分析报告或中学议论文？（严重扣分 -20）
- 缺少自嘲？（扣分 -10）
- 缺少"我"的第一人称视角？（扣分 -10）
- 结尾是愤怒或焦虑的？（扣分 -10）
- 出现 emoji 或表情符号？（扣分 -15）

你必须严格按以下 JSON 格式输出，不要输出其他内容：
{
  "style_score": 0-100,
  "content_score": 0-100,
  "structure_score": 0-100,
  "problems": ["问题1", "问题2"],
  "suggestions": "具体的修改建议"
}"""


def critique_article(article: str, topic: str, article_type: str) -> CriticResult:
    """评分文章，返回 CriticResult。"""
    type_name = ARTICLE_TYPES.get(article_type, "社会观察型")

    user = f"""请评分以下文章：

**写作主题**：{topic}
**目标类型**：{article_type} - {type_name}

--- 文章开始 ---
{article}
--- 文章结束 ---

请严格按 JSON 格式输出评分结果。"""

    raw = call_ollama(
        user, system=CRITIC_SYSTEM,
        model=_config["critic_model"],
        think=True,
        temperature=0.2,
    )

    if not raw:
        return _default_critic_result("模型返回为空")

    # 尝试多种方式提取 JSON
    json_str = _extract_json(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            style = _clamp_score(data.get("style_score", 60))
            content = _clamp_score(data.get("content_score", 60))
            structure = _clamp_score(data.get("structure_score", 60))
            problems = data.get("problems", [])
            suggestions = data.get("suggestions", "")
            avg_score = (style + content + structure) // 3

            return CriticResult(
                score=avg_score,
                style_score=style,
                content_score=content,
                structure_score=structure,
                problems=problems if isinstance(problems, list) else [str(problems)],
                suggestions=str(suggestions),
                passed=avg_score >= _config["score_threshold"],
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return _default_critic_result(f"JSON 解析失败: {e}")

    return _default_critic_result("未能从回复中提取 JSON")


def _extract_json(text: str) -> Optional[str]:
    """从模型输出中提取 JSON 字符串。"""
    # 方法 1：匹配包含 style_score 的 JSON 块
    m = re.search(r'\{[^{}]*"style_score"[^{}]*\}', text, re.DOTALL)
    if m:
        return m.group()

    # 方法 2：匹配 ``` 代码块中的 JSON
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        return m.group(1)

    # 方法 3：匹配任何 {...} 块
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        return m.group()

    return None


def _clamp_score(val) -> int:
    """确保分数在 0-100 范围内。"""
    try:
        return max(0, min(100, int(val)))
    except (ValueError, TypeError):
        return 60


def _default_critic_result(reason: str) -> CriticResult:
    """评分失败时的默认结果。"""
    print(f"[警告] 评分异常：{reason}，使用默认分数")
    return CriticResult(
        score=65,
        style_score=65,
        content_score=65,
        structure_score=65,
        problems=[reason],
        suggestions="建议检查文章是否有足够的自嘲、比喻、短句节奏",
        passed=False,
    )


# ============================================================
# Step 5: 去 AI 化审查
# ============================================================

DE_AI_SYSTEM = """你是一位文字编辑，专门识别和去除 AI 生成文本的痕迹。

需要检查并修正的 AI 写作痕迹：

【内容层面】
- 夸大象征意义："这标志着""作为...的证明" → 删除，直接说事实
- 宣传式语言："充满活力的""令人叹为观止的" → 换为具体描述
- 模糊归因："专家认为""业内人士指出" → 给具体来源或删除
- 公式化结尾："尽管面临挑战，未来依然光明" → 用克制或自嘲替代

【语言层面】
- AI 高频词：此外、至关重要、深入探讨、强调、格局、培养、增强 → 替换为：其实、无非、不过、大概
- 三段式滥用："无缝、直观和强大" → 改为两项或特色三段递进
- 否定式排比："这不仅仅是...更是..." → 直接说是什么
- 同义词循环：避免刻意换词
- 句式长度一致 → 长短交错

【风格层面】
- 粗体滥用 → 正文不用粗体
- 列表化表达 → 改为段落式叙述
- 过度限定 → 用"大概""多半""我猜"
- 通用积极结论 → "我也不知道"
- 谄媚语气 → 删除

你的任务：接收一篇文章，输出修正后的完整文章。只修正 AI 痕迹，保留文章的核心内容和风格。
第一行输出标题，空一行后输出正文。不要输出任何说明。"""


def de_ai_review(title: str, article: str) -> Tuple[str, str]:
    """对文章进行去 AI 化审查，返回修正后的 (title, article)。"""
    prompt = f"""请对以下文章进行去 AI 化修正，去除所有 AI 写作痕迹，但保留和菜头的写作风格。

--- 文章开始 ---
{title}

{article}
--- 文章结束 ---

请输出修正后的完整文章。第一行是标题，空一行后是正文。"""

    raw = call_ollama(prompt, system=DE_AI_SYSTEM,
                      model=_config["writer_model"],
                      think=False, temperature=0.5)

    if not raw or len(raw) < 100:
        return title, article  # 审查失败，返回原文

    lines = raw.strip().split("\n")
    new_title = ""
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            new_title = stripped
            content_start = i + 1
            break
    while content_start < len(lines) and not lines[content_start].strip():
        content_start += 1
    new_content = "\n".join(lines[content_start:]).strip()

    return (new_title or title), (new_content or article)


# ============================================================
# Step 6: 三稿比较
# ============================================================

COMPARE_SYSTEM = """你是一位严格的文学评论家。你需要比较三篇同一话题的文章，评选最佳稿，并列出所有缺陷。

评分维度（每项1-10分）：
1. 切入点吸引力：开头是否让人想继续读？
2. 论证说服力：逻辑是否自洽？比喻是否精准？
3. 风格还原度：多大程度上像和菜头本人写的？
4. 情感真实度：情感是流淌出来的还是强加的？
5. AI 痕迹残留：还有多少AI味？（10=完全没有AI味，1=全是AI味）

你必须严格按以下 JSON 格式输出，不要输出其他内容：
{
  "best_draft": "A" 或 "B" 或 "C",
  "scores": {
    "A": {"appeal": 0, "logic": 0, "style": 0, "emotion": 0, "anti_ai": 0, "total": 0},
    "B": {"appeal": 0, "logic": 0, "style": 0, "emotion": 0, "anti_ai": 0, "total": 0},
    "C": {"appeal": 0, "logic": 0, "style": 0, "emotion": 0, "anti_ai": 0, "total": 0}
  },
  "common_issues": ["所有草稿的共性问题1", "共性问题2"],
  "best_draft_issues": ["最佳稿特有的问题1"],
  "avoid_next_round": ["下一轮应规避的具体指令1", "指令2"]
}"""


@dataclass
class CompareResult:
    best_draft: str  # "A", "B", "C"
    scores: dict     # {A: {appeal, logic, style, emotion, anti_ai, total}, ...}
    common_issues: List[str]
    best_draft_issues: List[str]
    avoid_next_round: List[str]
    all_dimensions_pass: bool  # 所有维度 >= 7


def compare_three_drafts(
    drafts: List[Tuple[str, str]],  # [(title, article), ...]
    topic: str,
    article_type: str,
) -> CompareResult:
    """比较三篇草稿，返回 CompareResult。"""
    type_name = ARTICLE_TYPES.get(article_type, "社会观察型")

    prompt_parts = [f"请比较以下三篇关于「{topic}」的{type_name}文章：\n"]
    labels = ["A", "B", "C"]
    for i, (title, article) in enumerate(drafts):
        prompt_parts.append(f"--- 草稿 {labels[i]} ---\n{title}\n\n{article}\n")

    prompt_parts.append("\n请严格按 JSON 格式输出比较结果。")

    raw = call_ollama(
        "\n".join(prompt_parts), system=COMPARE_SYSTEM,
        model=_config["critic_model"],
        think=True, temperature=0.2,
    )

    if not raw:
        return _default_compare_result()

    json_str = _extract_json_deep(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            best = data.get("best_draft", "A")
            scores = data.get("scores", {})
            # 检查所有维度是否 >= 7
            all_pass = True
            if best in scores:
                for dim in ["appeal", "logic", "style", "emotion", "anti_ai"]:
                    if scores[best].get(dim, 0) < 7:
                        all_pass = False
                        break
            else:
                all_pass = False

            return CompareResult(
                best_draft=best,
                scores=scores,
                common_issues=data.get("common_issues", []),
                best_draft_issues=data.get("best_draft_issues", []),
                avoid_next_round=data.get("avoid_next_round", []),
                all_dimensions_pass=all_pass,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return _default_compare_result()


def _extract_json_deep(text: str) -> Optional[str]:
    """从模型输出中提取可能嵌套较深的 JSON。"""
    # 方法 1：匹配包含 best_draft 的 JSON 块
    m = re.search(r'\{[^{}]*"best_draft"[^}]*\}', text, re.DOTALL)
    if m:
        # 尝试向后扩展以包含完整的嵌套 JSON
        start = m.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break

    # 方法 2：匹配 ``` 代码块中的 JSON
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        return m.group(1)

    # 回退到基础提取
    return _extract_json(text)


def _default_compare_result() -> CompareResult:
    """比较失败时的默认结果。"""
    return CompareResult(
        best_draft="A",
        scores={"A": {"total": 30}, "B": {"total": 30}, "C": {"total": 30}},
        common_issues=["评分异常，使用默认结果"],
        best_draft_issues=[],
        avoid_next_round=["确保文章有具体切入点", "增加自嘲元素", "避免AI式的总结性语言"],
        all_dimensions_pass=False,
    )


# ============================================================
# 主工作流
# ============================================================

def run_workflow(request: ArticleRequest) -> WorkflowResult:
    """执行完整的写作工作流（v4.0：三稿并行 + 去AI化 + 迭代）。"""
    max_iterations = _config["max_iterations"]

    print("=" * 60)
    print("  和菜头风格自动写作工作流 v4.0")
    print("=" * 60)
    print(f"  主题：{request.topic}")
    print(f"  字数：{request.word_count}")
    print(f"  模型：{_config['writer_model']}")
    print(f"  上下文：{_config['num_ctx']} tokens")
    print(f"  参考策略：{_config['reference_mode']}")
    print(f"  模式：{'快速(单稿)' if request.quick_mode else '标准(三稿迭代)'}")
    print("=" * 60)

    # Step 0: 检查 Ollama
    print("\n[0/7] 检查 Ollama 服务...")
    if not check_ollama_ready():
        return WorkflowResult(status="error", article="", title="", article_type="")
    print(f"  OK 使用模型：{_config['writer_model']}")

    # Step 1: 事件分类（多类型叠加）
    if request.article_type:
        article_type = request.article_type.upper()
        secondary_type = request.secondary_type
        print(f"\n[1/7] 文章类型：{article_type} - "
              f"{ARTICLE_TYPES.get(article_type, '?')}（手动指定）")
        if secondary_type:
            print(f"  副类型：{secondary_type} - "
                  f"{ARTICLE_TYPES.get(secondary_type, '?')}")
    else:
        print("\n[1/7] 自动判断文章类型（允许多类型叠加）...")
        article_type, secondary_type = classify_event(request.topic)
        print(f"  → 主类型：{article_type} - {ARTICLE_TYPES.get(article_type, '?')}")
        if secondary_type:
            print(f"  → 副类型：{secondary_type} - {ARTICLE_TYPES.get(secondary_type, '?')}")
        else:
            print(f"  → 单一类型，无副类型")

    # Step 2: 检索参考原文
    print("\n[2/7] 检索参考原文...")
    all_articles = load_all_articles()

    if request.reference_article:
        anchor = find_article_by_title(all_articles, request.reference_article)
        if anchor:
            print(f"  → 锚点：《{anchor['title']}》({anchor['file']})")
            extra = search_articles(all_articles, request.topic,
                                    top_k=2, article_type=article_type)
            references = [anchor] + [a for a in extra if a != anchor][:2]
        else:
            print(f"  [警告] 未找到指定文章《{request.reference_article}》，使用自动检索")
            references = search_articles(all_articles, request.topic,
                                         top_k=3, article_type=article_type)
    else:
        references = search_articles(all_articles, request.topic,
                                     top_k=3, article_type=article_type)

    # 如有副类型，额外检索 1-2 篇
    if secondary_type and not request.reference_article:
        sec_refs = search_articles(all_articles, request.topic,
                                   top_k=2, article_type=secondary_type)
        for sr in sec_refs:
            if sr not in references:
                references.append(sr)
                if len(references) >= 5:
                    break

    ref_titles = [f"《{a['title']}》({a['file']})" for a in references]
    for t in ref_titles:
        print(f"  → {t}")

    # ===== 快速模式：单稿生成 + 自检 + 去AI化 =====
    if request.quick_mode:
        return _run_quick_workflow(
            request, article_type, secondary_type, references, ref_titles
        )

    # ===== 标准模式：三稿迭代 =====
    # 三篇草稿的差异维度
    DRAFT_VARIATIONS = [
        "切入角度独特：从一个出人意料的具体细节或场景切入，不要用最常见的角度",
        "论证路径独特：使用不同于常规的论证技法，如'追问到底'或'比喻穿刺'",
        "情感基调独特：整体偏冷幽默或偏温暖反思，与其他草稿形成差异",
    ]

    best_article = ""
    best_title = ""
    best_score = 0
    compare_history = []
    avoid_issues = []
    total_drafts = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"  第 {iteration}/{max_iterations} 轮迭代")
        if avoid_issues:
            print(f"  本轮需规避：{'; '.join(avoid_issues[:3])}...")
        print(f"{'=' * 60}")

        drafts = []  # [(title, article), ...]

        # Step 3: 三稿并行生成
        for draft_idx in range(3):
            label = ["A", "B", "C"][draft_idx]
            variation = DRAFT_VARIATIONS[draft_idx]

            print(f"\n[3/7] 生成草稿 {label}（{variation[:20]}...）")
            start_time = time.time()

            title, article = generate_article(
                topic=request.topic,
                article_type=article_type,
                word_count=request.word_count,
                reference_texts=references,
                style_hint=request.style_hint,
                secondary_type=secondary_type,
                draft_variation=variation,
                avoid_issues=avoid_issues if avoid_issues else None,
            )

            gen_time = time.time() - start_time
            print(f"  标题：{title} | {len(article)}字 | {gen_time:.0f}秒")

            # Step 4: 自检（50%通过）
            print(f"  [4/7] 自检校准（50%通过即可）...")
            critic = critique_article(article, request.topic, article_type)
            print(f"  评分：{critic.score} (风格{critic.style_score}/"
                  f"内容{critic.content_score}/结构{critic.structure_score})")

            # Step 5: 去 AI 化审查
            print(f"  [5/7] 去 AI 化审查...")
            title, article = de_ai_review(title, article)
            print(f"  去AI化完成：{len(article)}字")

            drafts.append((title, article))
            total_drafts += 1

        # Step 6: 三稿比较
        print(f"\n[6/7] 三稿比较...")
        compare = compare_three_drafts(drafts, request.topic, article_type)
        compare_history.append({
            "round": iteration,
            "best_draft": compare.best_draft,
            "scores": compare.scores,
            "common_issues": compare.common_issues,
        })

        best_idx = ["A", "B", "C"].index(compare.best_draft) if compare.best_draft in ["A", "B", "C"] else 0
        best_title_round, best_article_round = drafts[best_idx]

        # 计算当前轮的最佳分数
        current_scores = compare.scores.get(compare.best_draft, {})
        current_total = current_scores.get("total", 0)
        if not current_total and current_scores:
            # 计算总分
            current_total = sum(
                current_scores.get(d, 0)
                for d in ["appeal", "logic", "style", "emotion", "anti_ai"]
            )

        print(f"  最佳稿：草稿 {compare.best_draft}（总分 {current_total}）")
        if compare.common_issues:
            print(f"  共性问题：{'; '.join(compare.common_issues[:3])}")
        if compare.avoid_next_round:
            print(f"  下轮规避：{'; '.join(compare.avoid_next_round[:3])}")

        if current_total > best_score:
            best_score = current_total
            best_article = best_article_round
            best_title = best_title_round

        # Step 7: 判断是否提前终止
        if compare.all_dimensions_pass:
            print(f"\n  ✅ 所有维度 >= 7 分，提前结束迭代！")
            break

        if iteration < max_iterations:
            # 将缺陷清单传入下一轮
            avoid_issues = compare.avoid_next_round + compare.common_issues
            avoid_issues = avoid_issues[:6]  # 限制数量
            print(f"\n  → 继续第 {iteration + 1} 轮迭代...")

    type_str = f"{article_type} - {ARTICLE_TYPES.get(article_type, '')}"
    if secondary_type:
        type_str += f" + {secondary_type} - {ARTICLE_TYPES.get(secondary_type, '')}"

    status = "success" if best_score >= 35 else "max_iterations"

    return WorkflowResult(
        status=status,
        article=best_article,
        title=best_title,
        article_type=type_str,
        secondary_type=secondary_type,
        iterations=len(compare_history),
        total_drafts=total_drafts,
        final_score=best_score,
        reference_articles=ref_titles,
        compare_history=compare_history,
    )


def _run_quick_workflow(
    request: ArticleRequest,
    article_type: str,
    secondary_type: Optional[str],
    references: List[dict],
    ref_titles: List[str],
) -> WorkflowResult:
    """快速模式：只生成一篇，自检 + 去AI化后直接输出。"""
    print(f"\n[3/5] 快速模式：生成单篇文章...")
    start_time = time.time()

    title, article = generate_article(
        topic=request.topic,
        article_type=article_type,
        word_count=request.word_count,
        reference_texts=references,
        style_hint=request.style_hint,
        secondary_type=secondary_type,
    )
    gen_time = time.time() - start_time
    print(f"  标题：{title} | {len(article)}字 | {gen_time:.0f}秒")

    # 自检
    print(f"\n[4/5] 自检校准...")
    critic = critique_article(article, request.topic, article_type)
    print(f"  评分：{critic.score} (风格{critic.style_score}/"
          f"内容{critic.content_score}/结构{critic.structure_score})")

    # 去AI化
    print(f"\n[5/5] 去 AI 化审查...")
    title, article = de_ai_review(title, article)
    print(f"  完成：{len(article)}字")

    type_str = f"{article_type} - {ARTICLE_TYPES.get(article_type, '')}"
    if secondary_type:
        type_str += f" + {secondary_type} - {ARTICLE_TYPES.get(secondary_type, '')}"

    return WorkflowResult(
        status="success",
        article=article,
        title=title,
        article_type=type_str,
        secondary_type=secondary_type,
        iterations=1,
        total_drafts=1,
        final_score=critic.score,
        reference_articles=ref_titles,
        critic_history=[critic],
    )


# ============================================================
# 输出格式化
# ============================================================

def format_output(result: WorkflowResult) -> str:
    """格式化最终输出。"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  生成完成")
    lines.append("=" * 60)

    if result.status == "success":
        lines.append("状态：✅ 达标")
    elif result.status == "max_iterations":
        lines.append("状态：⚠️  达到最大迭代次数，取最佳版本")
    else:
        lines.append(f"状态：❌ {result.status}")

    lines.append(f"迭代：{result.iterations} 轮 | 总草稿数：{result.total_drafts}")
    lines.append(f"最终分数：{result.final_score}")
    lines.append(f"文章类型：{result.article_type}")
    lines.append(f"参考原文：{'、'.join(result.reference_articles)}")
    lines.append("")
    lines.append("─" * 60)
    lines.append(f"  {result.title}")
    lines.append("─" * 60)
    lines.append(result.article)
    lines.append("─" * 60)

    if result.compare_history:
        lines.append("\n[迭代历史]")
        for ch in result.compare_history:
            lines.append(f"  第{ch['round']}轮：最佳草稿 {ch['best_draft']}")
            if ch.get('common_issues'):
                lines.append(f"    共性问题：{'; '.join(ch['common_issues'][:2])}")

    if result.critic_history:
        lines.append("\n[评分历史]")
        for i, c in enumerate(result.critic_history, 1):
            lines.append(
                f"  第{i}轮：{c.score}分 "
                f"(风格{c.style_score} / 内容{c.content_score} / 结构{c.structure_score})"
            )

    return "\n".join(lines)


def save_output(result: WorkflowResult, output_dir: Path) -> Path:
    """将结果保存为 md 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r'[^\w\u4e00-\u9fff]', '_', result.title)[:30]
    filename = f"{timestamp}_{safe_title}.md"
    filepath = output_dir / filename

    # 评分历史
    score_lines = []
    for i, c in enumerate(result.critic_history, 1):
        score_lines.append(
            f"| 第{i}轮 | {c.style_score} | {c.content_score} "
            f"| {c.structure_score} | **{c.score}** |"
        )
    score_table = ""
    if score_lines:
        score_table = (
            "\n## 评分记录\n\n"
            "| 轮次 | 风格 | 内容 | 结构 | 综合 |\n"
            "|------|------|------|------|------|\n"
            + "\n".join(score_lines)
        )

    content = f"""# {result.title}

> **自动生成** | 类型：{result.article_type} | 最终分数：{result.final_score} | 迭代：{result.iterations}轮 | 总草稿数：{result.total_drafts}
> 参考原文：{'、'.join(result.reference_articles)}
> 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}
> 模型：{_config['writer_model']} | 上下文：{_config['num_ctx']}

---

{result.article}

---
{score_table}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[保存] {filepath}")
    return filepath


# ============================================================
# 交互模式
# ============================================================

def interactive_mode():
    """交互式写作模式，支持反复调整。"""
    print("=" * 60)
    print("  和菜头风格自动写作 - 交互模式")
    print("=" * 60)
    print()
    print("  命令：")
    print("    直接输入话题开始写作（三稿迭代模式）")
    print("    /quick 话题   快速模式（单稿，跳过三稿比较）")
    print("    /type A-F    指定文章类型（可指定两个如 /type C B）")
    print("    /words N     设置字数")
    print("    /style xxx   设置风格微调")
    print("    /ref 文章名   指定锚点原文")
    print("    /model xxx   切换模型")
    print("    /ctx N       设置上下文窗口")
    print("    /iter N      设置最大迭代次数")
    print("    /mode full|smart 参考原文策略")
    print("    /search 关键词  搜索原文库")
    print("    /config      显示当前配置")
    print("    /help        显示帮助")
    print("    /quit        退出")
    print()

    # 检查 Ollama
    if not check_ollama_ready():
        print("\n请先启动 Ollama 再使用交互模式。")
        return

    # 预加载原文
    articles = load_all_articles()

    # 交互状态
    current_type = None
    current_secondary_type = None
    current_words = _config["default_word_count"]
    current_style = None
    current_ref = None
    current_quick = False

    while True:
        try:
            user_input = input("\n[话题] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                print("再见！")
                break

            elif cmd == "/type":
                types = arg.upper().split()
                if types and types[0] in ARTICLE_TYPES:
                    current_type = types[0]
                    current_secondary_type = types[1] if len(types) > 1 and types[1] in ARTICLE_TYPES else None
                    print(f"  主类型 → {current_type} - {ARTICLE_TYPES[current_type]}")
                    if current_secondary_type:
                        print(f"  副类型 → {current_secondary_type} - {ARTICLE_TYPES[current_secondary_type]}")
                elif arg.lower() == "auto":
                    current_type = None
                    current_secondary_type = None
                    print("  文章类型 → 自动判断（允许多类型叠加）")
                else:
                    print(f"  可选类型：{', '.join(f'{k}={v}' for k, v in ARTICLE_TYPES.items())}")
                    print(f"  支持双类型，如：/type C B（主C + 副B）")

            elif cmd == "/words":
                try:
                    current_words = int(arg)
                    print(f"  字数 → {current_words}")
                except ValueError:
                    print("  请输入数字，如 /words 2000")

            elif cmd == "/style":
                if arg:
                    current_style = arg
                    print(f"  风格 → {current_style}")
                else:
                    current_style = None
                    print("  风格 → 默认")

            elif cmd == "/ref":
                if arg:
                    found = find_article_by_title(articles, arg)
                    if found:
                        current_ref = found["title"]
                        print(f"  锚点原文 → 《{current_ref}》({found['file']})")
                    else:
                        print(f"  未找到《{arg}》")
                        # 模糊搜索建议
                        suggestions = search_articles(articles, arg, top_k=5)
                        if suggestions:
                            print("  你是否想找：")
                            for s in suggestions:
                                print(f"    - {s['title']} ({s['file']})")
                else:
                    current_ref = None
                    print("  锚点原文 → 无（自动检索）")

            elif cmd == "/model":
                if arg:
                    _config["writer_model"] = arg
                    _config["critic_model"] = arg
                    print(f"  模型 → {arg}")
                else:
                    print(f"  当前模型：{_config['writer_model']}")

            elif cmd == "/ctx":
                try:
                    _config["num_ctx"] = int(arg)
                    print(f"  上下文 → {_config['num_ctx']} tokens")
                except ValueError:
                    print(f"  当前上下文：{_config['num_ctx']} tokens")

            elif cmd == "/iter":
                try:
                    _config["max_iterations"] = int(arg)
                    print(f"  最大迭代 → {_config['max_iterations']}")
                except ValueError:
                    print(f"  当前最大迭代：{_config['max_iterations']}")

            elif cmd == "/quick":
                if arg:
                    # /quick 后面直接跟话题，快速生成
                    current_quick = True
                    print("  已切换为快速模式（本次生效）")
                    # 将话题传入后续处理
                    user_input = arg
                    # 不 continue，让后续代码处理
                else:
                    current_quick = not current_quick
                    print(f"  快速模式 → {'开启' if current_quick else '关闭'}")
                    continue

            elif cmd == "/mode":
                if arg in ("full", "smart"):
                    _config["reference_mode"] = arg
                    print(f"  参考策略 → {arg}")
                else:
                    print(f"  当前策略：{_config['reference_mode']}（可选：full, smart）")

            elif cmd == "/search":
                if arg:
                    results = search_articles(articles, arg, top_k=10)
                    print(f"  搜索 \"{arg}\" 找到 {len(results)} 篇：")
                    for r in results:
                        preview = r["content"][:60].replace("\n", " ")
                        print(f"    [{r['file']}] {r['title']} — {preview}...")
                else:
                    print("  用法：/search 关键词")

            elif cmd == "/config":
                print("  当前配置：")
                for k, v in _config.items():
                    print(f"    {k}: {v}")
                print(f"    当前主类型: {current_type or '自动'}")
                print(f"    当前副类型: {current_secondary_type or '无'}")
                print(f"    当前字数: {current_words}")
                print(f"    当前风格: {current_style or '默认'}")
                print(f"    锚点原文: {current_ref or '无'}")
                print(f"    快速模式: {'开启' if current_quick else '关闭'}")

            elif cmd == "/help":
                print("  命令列表：")
                print("    /type A-F|auto  设置/自动判断文章类型（支持双类型如 /type C B）")
                print("    /words N        设置字数")
                print("    /style xxx      设置风格微调（更毒舌/更佛系/更幽默）")
                print("    /ref 文章名      指定锚点原文")
                print("    /quick [话题]    快速模式（单稿，跳过三稿比较）")
                print("    /model xxx      切换 Ollama 模型")
                print("    /ctx N          设置上下文窗口大小")
                print("    /iter N         设置最大迭代次数")
                print("    /mode full|smart 参考原文策略")
                print("    /search 关键词   搜索原文库")
                print("    /config         显示当前配置")
                print("    /quit           退出")
            else:
                print(f"  未知命令：{cmd}，输入 /help 查看帮助")

            continue

        # 正常输入 → 开始写作
        request = ArticleRequest(
            topic=user_input,
            word_count=current_words,
            article_type=current_type,
            secondary_type=current_secondary_type,
            style_hint=current_style,
            reference_article=current_ref,
            quick_mode=current_quick,
        )

        result = run_workflow(request)
        print(format_output(result))

        # 保存
        output_dir = SCRIPT_DIR / "output"
        save_output(result, output_dir)

        # 写作完成后重置一次性设置
        current_ref = None  # 锚点只用一次
        if current_quick and cmd == "/quick":
            current_quick = False  # /quick 话题 模式只用一次


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="和菜头风格自动写作工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

  # 自动判断类型并生成（三稿迭代模式）
  python hecaitou_writer.py --topic "最近大家都在用AI写东西"

  # 快速模式（只生成一篇，跳过三稿比较）
  python hecaitou_writer.py --topic "最近大家都在用AI写东西" --quick

  # 指定主类型 + 副类型（融合两种风格）
  python hecaitou_writer.py --topic "某AI公司创始人猝死" --type C --type2 B

  # 指定字数和风格微调
  python hecaitou_writer.py --topic "苹果发布新AI" --type B --style "更毒舌一些"

  # 指定锚点原文
  python hecaitou_writer.py --topic "网红带货翻车" --ref "松鼠尾巴"

  # 使用完整原文作为参考（需要更大上下文）
  python hecaitou_writer.py --topic "xxx" --mode full --ctx 32768

  # 交互模式（推荐）
  python hecaitou_writer.py --interactive

文章类型（允许主类型 + 副类型叠加）：
  A = 社会观察型    B = 技术/产品体验型    C = 生死/无常型
  D = 自省/修行型    E = 文化/阅读型        F = 日常生活型

工作流程（标准模式）：
  Step 1：事件分类（允许多类型） → Step 2：检索参考原文
  Step 3：三稿并行生成 → Step 4：自检(50%通过) → Step 5：去AI化
  Step 6：三稿比较 → Step 7：迭代优化（最多3轮）

参考原文策略：
  smart = 根据上下文窗口自动裁剪（默认，省内存）
  full  = 使用完整原文（需要更大 num_ctx）
        """,
    )

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--topic", "-t", help="写作主题/事件描述")
    mode_group.add_argument("--interactive", "-i", action="store_true",
                            help="交互模式（推荐）")

    # 写作参数
    parser.add_argument("--words", "-w", type=int,
                        default=_config["default_word_count"],
                        help=f"目标字数（默认 {_config['default_word_count']}）")
    parser.add_argument("--type", choices=list("ABCDEF"), default=None,
                        help="主文章类型（不指定则自动判断）")
    parser.add_argument("--type2", choices=list("ABCDEF"), default=None,
                        help="副文章类型（可选，融合两种风格）")
    parser.add_argument("--style", "-s", default=None,
                        help="风格微调，如 '更毒舌' '更佛系' '更幽默'")
    parser.add_argument("--ref", default=None,
                        help="指定锚点原文标题")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="快速模式：只生成一篇，跳过三稿比较")

    # 模型参数
    parser.add_argument("--model", "-m", default=None,
                        help=f"Ollama 模型名（默认 {_config['writer_model']}）")
    parser.add_argument("--ctx", type=int, default=None,
                        help=f"上下文窗口大小（默认 {_config['num_ctx']}）")

    # 工作流参数
    parser.add_argument("--max-iter", type=int, default=None,
                        help=f"最大迭代轮数（默认 {_config['max_iterations']}）")
    parser.add_argument("--mode", choices=["full", "smart"], default=None,
                        help="参考原文策略（默认 smart）")

    # 输出参数
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录（默认 ./output/）")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存到文件，只打印")

    args = parser.parse_args()

    # 应用配置
    if args.model:
        _config["writer_model"] = args.model
        _config["critic_model"] = args.model
    if args.ctx is not None:
        _config["num_ctx"] = args.ctx
    if args.max_iter is not None:
        _config["max_iterations"] = args.max_iter
    if args.mode:
        _config["reference_mode"] = args.mode

    # 交互模式
    if args.interactive:
        interactive_mode()
        return

    # 命令行模式
    if not args.topic:
        parser.print_help()
        print("\n请使用 --topic 指定话题，或使用 --interactive 进入交互模式。")
        return

    request = ArticleRequest(
        topic=args.topic,
        word_count=args.words,
        article_type=args.type,
        secondary_type=args.type2,
        style_hint=args.style,
        reference_article=args.ref,
        quick_mode=args.quick,
    )

    result = run_workflow(request)
    print(format_output(result))

    if not args.no_save:
        output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output"
        save_output(result, output_dir)


if __name__ == "__main__":
    main()
