#!/usr/bin/env python3
"""
将1072篇和菜头文章按六大类型 + 子分类重新整理为6个Markdown文件。
分类基于关键词匹配 + 内容特征分析，不依赖LLM。
"""

import re
import glob
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
PASSAGES_DIR = SCRIPT_DIR / "passages"
OUTPUT_DIR = SCRIPT_DIR / "classified"

# ============================================================
# 六大类型定义 + 子分类 + 关键词规则
# ============================================================

CATEGORIES = {
    "A": {
        "name": "社会观察型",
        "file": "A_社会观察.md",
        "subcategories": {
            "A1_网络现象与舆论": {
                "keywords": ["舆论", "热搜", "网暴", "键盘侠", "喷子", "吃瓜", "反转",
                             "公众号", "流量", "网红", "直播", "带货", "粉丝", "up主",
                             "短视频", "抖音", "微博", "朋友圈", "社交媒体", "互联网",
                             "评论区", "留言", "私信", "后台", "读者", "订阅",
                             "P图", "照片", "自拍", "打卡", "人设", "营销号"],
                "title_keywords": ["网", "舆", "流量", "粉丝", "网红"],
            },
            "A2_社会新闻与公共事件": {
                "keywords": ["新闻", "事件", "案件", "判决", "法院", "警察", "政府",
                             "政策", "规定", "禁止", "罚款", "拆迁", "维权",
                             "房价", "房贷", "物业", "业主", "小区", "城市",
                             "农村", "打工", "外卖", "快递", "996", "加班",
                             "裁员", "失业", "就业", "考公", "考研", "内卷"],
                "title_keywords": ["房", "城", "打工", "裁", "考"],
            },
            "A3_教育与代际": {
                "keywords": ["教育", "学校", "老师", "学生", "孩子", "家长", "高考",
                             "补课", "辅导", "鸡娃", "虎妈", "早恋", "叛逆",
                             "年轻人", "老年人", "父母", "儿子", "女儿", "催婚",
                             "生育", "结婚", "离婚", "单身", "相亲", "恋爱",
                             "代际", "一代人", "两代人", "上一代", "下一代"],
                "title_keywords": ["教育", "孩子", "父", "母", "婚", "恋"],
            },
            "A4_人性与群体行为": {
                "keywords": ["人性", "自私", "贪婪", "虚伪", "善良", "恶意",
                             "嫉妒", "攀比", "面子", "虚荣", "焦虑", "恐惧",
                             "从众", "站队", "撕裂", "对立", "偏见", "歧视",
                             "标签", "刻板印象", "共情", "同理心",
                             "聪明人", "蠢人", "好人", "坏人", "普通人",
                             "大众", "群体", "社会", "世界", "时代"],
                "title_keywords": ["人", "世", "时代"],
            },
            "A5_名人与公众人物": {
                "keywords": ["明星", "名人", "公众人物", "企业家", "作家", "导演",
                             "歌手", "演员", "主持人", "网络大V", "博主",
                             "马云", "马斯克", "罗永浩", "罗振宇", "许知远",
                             "韩寒", "郭德纲", "周杰伦", "李诞", "俞敏洪",
                             "川普", "特朗普", "拜登", "奥巴马"],
                "title_keywords": [],
            },
        },
    },
    "B": {
        "name": "技术/产品体验型",
        "file": "B_技术产品.md",
        "subcategories": {
            "B1_AI与人工智能": {
                "keywords": ["AI", "人工智能", "ChatGPT", "GPT", "大模型", "语言模型",
                             "Midjourney", "DALL", "Stable Diffusion", "Sora",
                             "机器学习", "深度学习", "神经网络", "算法", "训练",
                             "Prompt", "生成式", "AIGC", "Gemini", "Claude",
                             "OpenAI", "百度", "通义", "文心", "豆包"],
                "title_keywords": ["AI", "人工智能", "GPT", "模型", "Midjourney"],
            },
            "B2_数码产品与消费": {
                "keywords": ["苹果", "iPhone", "iPad", "Mac", "Apple",
                             "安卓", "手机", "电脑", "笔记本", "平板",
                             "耳机", "音箱", "相机", "镜头", "NAS",
                             "充电", "电池", "屏幕", "芯片", "处理器",
                             "购买", "价格", "性价比", "开箱", "体验",
                             "HiFi", "音响", "耳放", "解码器", "DAC"],
                "title_keywords": ["苹果", "Apple", "iPhone", "NAS", "HiFi"],
            },
            "B3_互联网行业与产品": {
                "keywords": ["互联网", "创业", "融资", "上市", "市值",
                             "产品", "用户", "体验", "设计", "功能",
                             "微信", "支付宝", "淘宝", "京东", "拼多多",
                             "百度", "腾讯", "阿里", "字节", "美团",
                             "谷歌", "Google", "微软", "亚马逊", "Facebook",
                             "App", "软件", "平台", "算法", "推荐"],
                "title_keywords": ["Google", "谷歌", "微信", "库克", "创始人"],
            },
            "B4_科技趋势与思考": {
                "keywords": ["技术", "科技", "创新", "未来", "趋势",
                             "数字化", "自动化", "区块链", "元宇宙", "虚拟",
                             "数据", "隐私", "监控", "安全", "加密"],
                "title_keywords": ["技术", "科技", "未来"],
            },
        },
    },
    "C": {
        "name": "生死/无常型",
        "file": "C_生死无常.md",
        "subcategories": {
            "C1_死亡与告别": {
                "keywords": ["死亡", "去世", "离世", "逝世", "离开", "告别",
                             "葬礼", "墓地", "骨灰", "遗体", "追悼",
                             "永远", "再也不", "最后一次", "来不及",
                             "R.I.P", "走了", "没了"],
                "title_keywords": ["死", "别", "走", "逝"],
            },
            "C2_疾病与健康": {
                "keywords": ["疾病", "病", "手术", "医院", "医生", "治疗",
                             "癌症", "肿瘤", "心脏", "血压", "血糖",
                             "他汀", "药物", "体检", "检查", "指标",
                             "疫情", "新冠", "感染", "核酸", "疫苗",
                             "健康", "身体", "免疫", "康复", "病房"],
                "title_keywords": ["病", "医", "健康", "他汀", "疫"],
            },
            "C3_无常与苦难": {
                "keywords": ["无常", "苦难", "灾难", "地震", "洪水", "事故",
                             "意外", "不幸", "命运", "宿命", "劫数",
                             "痛苦", "悲伤", "绝望", "崩溃", "承受"],
                "title_keywords": ["无常", "苦", "难", "灾"],
            },
            "C4_中年与衰老": {
                "keywords": ["中年", "衰老", "老了", "年纪", "岁数",
                             "白发", "皱纹", "体力", "精力", "记忆力",
                             "退休", "养老", "晚年", "余生"],
                "title_keywords": ["中年", "老", "衰"],
            },
        },
    },
    "D": {
        "name": "自省/修行型",
        "file": "D_自省修行.md",
        "subcategories": {
            "D1_佛学与修行": {
                "keywords": ["佛", "佛学", "佛教", "佛法", "菩萨", "观音",
                             "禅", "禅修", "冥想", "打坐", "念经",
                             "因果", "业力", "轮回", "涅槃", "解脱",
                             "仁波切", "宗萨", "活佛", "上师", "喇嘛",
                             "缘起", "无明", "执着", "放下", "慈悲",
                             "修行", "修炼", "功课", "发愿", "回向"],
                "title_keywords": ["佛", "修行", "缘起", "因果", "比喻"],
            },
            "D2_个人反思与习惯": {
                "keywords": ["反思", "反省", "回顾", "总结", "复盘",
                             "习惯", "自律", "改变", "坚持", "放弃",
                             "决心", "计划", "目标", "清单",
                             "早起", "运动", "跑步", "减肥", "戒",
                             "发现", "意识到", "明白了", "原来"],
                "title_keywords": ["反思", "习惯", "发现"],
            },
            "D3_价值观与人生态度": {
                "keywords": ["价值", "意义", "目的", "追求", "选择",
                             "自由", "独立", "孤独", "寂寞", "独处",
                             "简单", "极简", "断舍离", "够用",
                             "认知", "智慧", "清醒", "糊涂",
                             "接受", "承认", "面对", "不争", "不辩"],
                "title_keywords": ["有所不为", "选择", "自由", "孤独"],
            },
            "D4_年终总结与阶段回顾": {
                "keywords": ["年终", "年度", "总结", "回顾", "盘点",
                             "这一年", "过去一年", "新年", "跨年",
                             "生日", "周年", "纪念"],
                "title_keywords": ["年", "总结", "回顾", "告别"],
            },
        },
    },
    "E": {
        "name": "文化/阅读型",
        "file": "E_文化阅读.md",
        "subcategories": {
            "E1_读书与写作": {
                "keywords": ["读书", "阅读", "书", "小说", "作者", "作家",
                             "文学", "诗", "散文", "写作", "文字", "文章",
                             "出版", "编辑", "版本", "翻译",
                             "公众号", "博客", "写了", "发了"],
                "title_keywords": ["读", "书", "写", "文"],
            },
            "E2_电影音乐与艺术": {
                "keywords": ["电影", "影片", "导演", "演员", "票房",
                             "音乐", "歌", "专辑", "乐队", "爵士",
                             "画", "绘画", "摄影", "展览", "美术馆",
                             "艺术", "审美", "设计", "创作"],
                "title_keywords": ["电影", "音乐", "歌", "画", "艺术"],
            },
            "E3_历史与文化": {
                "keywords": ["历史", "古代", "朝代", "皇帝", "大臣",
                             "文化", "传统", "民族", "文明",
                             "哲学", "思想", "儒", "道", "墨",
                             "故事", "传说", "典故", "成语"],
                "title_keywords": ["历史", "文化", "古", "传"],
            },
            "E4_语言与翻译": {
                "keywords": ["语言", "语法", "词汇", "表达", "翻译",
                             "英语", "中文", "外语", "方言",
                             "语感", "修辞", "比喻"],
                "title_keywords": ["翻译", "语言", "词"],
            },
        },
    },
    "F": {
        "name": "日常生活型",
        "file": "F_日常生活.md",
        "subcategories": {
            "F1_美食与烹饪": {
                "keywords": ["食物", "美食", "做饭", "烹饪", "厨房",
                             "餐厅", "饭店", "外卖", "点菜",
                             "好吃", "难吃", "味道", "口感", "鲜",
                             "火锅", "面条", "饺子", "米饭", "面包",
                             "咖啡", "茶", "酒", "啤酒", "威士忌",
                             "菜", "肉", "鱼", "鸡", "牛", "猪",
                             "辣", "甜", "酸", "苦", "咸",
                             "小笼包", "烧烤", "串"],
                "title_keywords": ["吃", "食", "喝", "酒", "咖啡", "茶", "菜", "包"],
            },
            "F2_猫与宠物": {
                "keywords": ["猫", "喵", "猫咪", "铲屎", "猫粮", "猫砂",
                             "宠物", "狗", "毛孩子",
                             "撸猫", "吸猫", "猫主子"],
                "title_keywords": ["猫"],
            },
            "F3_旅行与见闻": {
                "keywords": ["旅行", "旅游", "出发", "出门", "回来",
                             "飞机", "高铁", "火车", "酒店", "民宿",
                             "风景", "景点", "古镇", "海边", "山",
                             "云南", "北京", "上海", "深圳", "广州",
                             "大理", "丽江", "拉萨", "西藏",
                             "日本", "泰国", "欧洲", "美国"],
                "title_keywords": ["旅", "游", "行", "回"],
            },
            "F4_日常琐事与季节": {
                "keywords": ["天气", "下雨", "下雪", "太阳", "春天", "夏天",
                             "秋天", "冬天", "季节", "温度",
                             "睡觉", "午睡", "失眠", "早起", "起床",
                             "散步", "逛街", "买", "快递", "收到",
                             "搬家", "装修", "打扫", "整理"],
                "title_keywords": ["天", "睡", "春", "夏", "秋", "冬"],
            },
            "F5_个人生活与感悟": {
                "keywords": ["生活", "日子", "平淡", "快乐", "幸福",
                             "无聊", "有趣", "好玩", "舒服",
                             "朋友", "聚会", "聊天", "电话"],
                "title_keywords": ["生活", "快乐", "日"],
            },
        },
    },
}


# ============================================================
# 文章加载
# ============================================================

def load_all_articles() -> List[dict]:
    """加载所有原文。"""
    articles = []
    md_patterns = [
        str(PASSAGES_DIR / "*.md"),
        str(SCRIPT_DIR / "20??-??.md"),
    ]
    md_files = []
    for pattern in md_patterns:
        md_files.extend(glob.glob(pattern))
    md_files = sorted(set(md_files))

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
            })

    return articles


# ============================================================
# 分类引擎
# ============================================================

def classify_article(article: dict) -> Tuple[str, str, float]:
    """
    对一篇文章进行分类。
    返回 (主类型代码, 子分类代码, 置信度)。
    置信度越高说明匹配越确定。
    """
    title = article["title"]
    # 取正文（去掉元数据部分）
    content = article["content"]
    # 去掉开头的元数据行
    lines = content.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]) if body_start > 0 else content
    # 只用前2000字做分类（够了）
    text = title + " " + body[:2000]
    text_lower = text.lower()

    best_cat = "A"
    best_subcat = "A4_人性与群体行为"  # 默认
    best_score = 0.0

    for cat_code, cat_info in CATEGORIES.items():
        for subcat_code, subcat_info in cat_info["subcategories"].items():
            score = 0.0

            # 正文关键词匹配
            for kw in subcat_info["keywords"]:
                count = text_lower.count(kw.lower())
                if count > 0:
                    score += count * 1.0
                    # 高频词降权
                    if count > 5:
                        score += 2.0  # 封顶额外加分

            # 标题关键词匹配（权重5x）
            for kw in subcat_info.get("title_keywords", []):
                if kw.lower() in title.lower():
                    score += 5.0

            if score > best_score:
                best_score = score
                best_cat = cat_code
                best_subcat = subcat_code

    return best_cat, best_subcat, best_score


# ============================================================
# 输出生成
# ============================================================

def extract_body(content: str) -> str:
    """从文章内容中提取纯正文（去掉元数据和图片链接）。"""
    lines = content.split("\n")
    body_lines = []
    started = False

    for line in lines:
        # 跳过元数据行
        if line.startswith("**日期**:") or line.startswith("---"):
            started = True
            continue
        # 跳过图片链接
        if re.match(r'^\[?\]?\(https?://.*\)$', line.strip()):
            continue
        if re.match(r'^!\[.*\]\(.*\)$', line.strip()):
            continue
        # 跳过空图片引用
        if line.strip().startswith('[](http'):
            continue
        if started or not line.startswith("**"):
            body_lines.append(line)

    result = "\n".join(body_lines).strip()
    # 清理多余空行
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


def generate_classified_files(articles: List[dict]):
    """生成6个分类Markdown文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 分类所有文章
    classified = defaultdict(lambda: defaultdict(list))
    unclassified_count = 0

    for art in articles:
        cat, subcat, score = classify_article(art)
        if score < 1.0:
            # 低置信度，放到A4（人性与群体行为）作为兜底
            cat = "A"
            subcat = "A4_人性与群体行为"
            unclassified_count += 1
        classified[cat][subcat].append((art, score))

    # 统计
    print("\n" + "=" * 60)
    print("  分类统计")
    print("=" * 60)
    total = 0
    for cat_code in sorted(CATEGORIES.keys()):
        cat_info = CATEGORIES[cat_code]
        cat_total = sum(len(v) for v in classified[cat_code].values())
        total += cat_total
        print(f"\n  {cat_code} {cat_info['name']}：{cat_total} 篇")
        for subcat_code in sorted(cat_info["subcategories"].keys()):
            count = len(classified[cat_code].get(subcat_code, []))
            if count > 0:
                subcat_name = subcat_code.split("_", 1)[1]
                print(f"    {subcat_code} {subcat_name}：{count} 篇")
    print(f"\n  总计：{total} 篇（低置信度兜底：{unclassified_count} 篇）")
    print("=" * 60)

    # 生成文件
    for cat_code in sorted(CATEGORIES.keys()):
        cat_info = CATEGORIES[cat_code]
        filepath = OUTPUT_DIR / cat_info["file"]

        lines = []
        lines.append(f"# {cat_code} {cat_info['name']}\n")
        lines.append(f"> 本文件收录和菜头「槽边往事」中属于**{cat_info['name']}**的文章。")
        lines.append(f"> 按子话题分类整理，便于风格模仿时快速检索参考原文。\n")

        cat_total = sum(len(v) for v in classified[cat_code].values())
        lines.append(f"**文章总数**：{cat_total} 篇\n")

        # 目录
        lines.append("---\n")
        lines.append("## 目录\n")
        for subcat_code in sorted(cat_info["subcategories"].keys()):
            items = classified[cat_code].get(subcat_code, [])
            if items:
                subcat_name = subcat_code.split("_", 1)[1]
                anchor = subcat_code.lower().replace("_", "-")
                lines.append(f"- [{subcat_code} {subcat_name}](#{anchor})（{len(items)} 篇）")
        lines.append("")

        # 各子分类
        for subcat_code in sorted(cat_info["subcategories"].keys()):
            items = classified[cat_code].get(subcat_code, [])
            if not items:
                continue

            subcat_name = subcat_code.split("_", 1)[1]
            lines.append(f"\n---\n")
            lines.append(f"## {subcat_code} {subcat_name}\n")
            lines.append(f"> {len(items)} 篇文章\n")

            # 按日期排序（新的在前）
            items.sort(key=lambda x: x[0].get("date", ""), reverse=True)

            for art, score in items:
                title = art["title"]
                date = art.get("date", "未知")
                body = extract_body(art["content"])

                lines.append(f"### {title}\n")
                lines.append(f"**日期**：{date} | **来源**：{art['file']}\n")
                lines.append(body)
                lines.append("")

        # 写文件
        content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        file_size = os.path.getsize(filepath)
        print(f"  [{cat_code}] {filepath.name} ({cat_total} 篇, {file_size/1024:.0f}KB)")

    print(f"\n文件已保存到：{OUTPUT_DIR}/")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("加载原文库...")
    articles = load_all_articles()
    print(f"共加载 {len(articles)} 篇文章\n")
    generate_classified_files(articles)
