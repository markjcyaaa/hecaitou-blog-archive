# 槽边往事博客爬虫 (hecaitou.com Blog Crawler)

一个用于爬取[和菜头的槽边往事](https://www.hecaitou.com/)博客全部文章的 Python 爬虫工具。爬取的文章按月份归档，每个月的文章保存为一个 Markdown 文件。

## 项目结构

```
.
├── crawler.py              # 爬虫主程序
├── requirements.txt        # Python 依赖
├── README.md               # 本说明文件
├── .gitignore              # Git 忽略规则
├── articles/               # [自动生成] 文章输出目录
│   ├── 2023-05.md          #   2023年5月的所有文章
│   ├── 2023-06.md          #   2023年6月的所有文章
│   ├── ...                 #   ...
│   └── 2026-03.md          #   2026年3月的所有文章
├── crawler.log             # [自动生成] 爬虫运行日志
└── .crawler_checkpoint.json # [自动生成] 断点续爬检查点
```

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd <repo-name>

# 2. (推荐) 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
# 完整爬取全部文章（预计需要 1-2 小时，取决于网络状况和文章总量）
python crawler.py

# 仅爬取某一年的文章
python crawler.py --year 2025

# 仅爬取某个月的文章
python crawler.py --month 2025-03

# 查看所有文章列表（不下载）
python crawler.py --list-only
```

## 详细使用说明

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--year` | 仅爬取指定年份的文章 | `--year 2025` |
| `--month` | 仅爬取指定月份的文章 | `--month 2025-03` |
| `--resume` | 从上次断点继续爬取 | `--resume` |
| `--list-only` | 仅列出文章URL，不下载 | `--list-only` |
| `--delay` | 请求间隔秒数（默认2秒） | `--delay 3` |
| `--output` | 输出目录（默认 `articles`） | `--output my_articles` |
| `--verbose` / `-v` | 显示详细调试信息 | `-v` |

### 典型用法

#### 1. 首次完整爬取

```bash
python crawler.py
```

这会：
1. 从 sitemap.xml 获取全部文章链接（约 1000+ 篇）
2. 逐个爬取每篇文章的标题、日期和正文
3. 按月份分组，每个月生成一个 Markdown 文件
4. 所有文件保存到 `articles/` 目录

**预计耗时**：取决于文章数量和网络延迟。以默认 2 秒间隔计算，1000 篇文章约需 35 分钟。

#### 2. 断点续爬

如果爬取过程中被中断（网络故障、手动中止等），可以使用 `--resume` 继续：

```bash
python crawler.py --resume
```

爬虫会自动跳过已经下载过的文章，从断点处继续。进度保存在 `.crawler_checkpoint.json` 文件中。

#### 3. 增量更新

如果博客发布了新文章，想只爬取最新月份：

```bash
# 只爬取当月
python crawler.py --month 2026-03

# 或者只爬取今年的
python crawler.py --year 2026
```

#### 4. 调整爬取速度

默认每次请求间隔 2 秒。如果遇到限流或想更保守：

```bash
# 更保守的爬取速度（5秒间隔）
python crawler.py --delay 5

# 更快（但可能被限流）
python crawler.py --delay 1
```

#### 5. 查看进度和日志

```bash
# 查看实时日志
tail -f crawler.log

# 查看爬取进度
cat .crawler_checkpoint.json | python -m json.tool | head -5
```

## 输出格式

每个月份的 Markdown 文件格式如下：

```markdown
# 槽边往事 - 2025年三月

> 本文件收录了 [2025年03月] 的所有文章，共 28 篇。
> 来源: https://www.hecaitou.com/
> 自动抓取时间: 2026-03-07 12:00:00

## 目录

1. [03-01] [文章标题1](#文章标题1)
2. [03-02] [文章标题2](#文章标题2)
...

---

## 文章标题1

**日期**: 2025-03-01T04:00:00Z | **原文链接**: [https://...](https://...)

---

这里是文章正文内容...

---

## 文章标题2
...
```

## 工作原理

### 爬取流程

```
┌─────────────────────────────┐
│ 1. 获取 sitemap.xml         │
│    (包含所有文章的URL和日期)  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. 解析 sitemap             │
│    提取所有 /YYYY/MM/*.html  │
│    URL，按日期排序           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. 逐个爬取文章页面          │
│    - 提取标题 (h3.post-title)│
│    - 提取日期 (abbr/time)    │
│    - 提取正文 (.post-body)   │
│    - HTML → Markdown 转换    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. 按月份分组保存            │
│    articles/YYYY-MM.md       │
│    每个文件包含当月所有文章    │
└─────────────────────────────┘
```

### 关键技术细节

- **数据来源**：使用博客的 `sitemap.xml` 获取完整文章列表，确保不遗漏
- **URL格式**：`https://www.hecaitou.com/YYYY/MM/slug.html`
- **内容解析**：使用 BeautifulSoup 解析 Blogger 模板的标准 HTML 结构
- **Markdown转换**：使用 markdownify 库将 HTML 转为干净的 Markdown
- **断点续爬**：通过 JSON 检查点文件记录已完成的 URL
- **错误重试**：每个请求最多重试 3 次，重试间隔递增
- **请求限流**：默认 2 秒间隔，防止对目标服务器造成压力

## 注意事项

### 礼貌爬取

- 请勿将 `--delay` 设置为 0，尊重目标服务器
- 建议使用默认的 2 秒间隔
- 如遇到 429 (Too Many Requests) 错误，请增大 delay 值

### 内容版权

- 爬取的文章内容版权归原作者（和菜头）所有
- 本工具仅供个人学习、备份和离线阅读使用
- 请勿将爬取的内容用于商业用途或未经授权的再发布

### 已知限制

- 文章中的图片保留为原始外链（Blogger CDN 地址），不下载到本地
- 某些包含特殊字符的 URL 可能需要手动处理
- 极早期文章（2016年等）的 HTML 结构可能与新文章略有不同

## 常见问题

### Q: 爬取中途网络断了怎么办？

爬虫会自动保存进度。重新运行时加上 `--resume` 参数即可从断点继续：

```bash
python crawler.py --resume
```

### Q: 如何只更新最新的文章？

按月份增量爬取：

```bash
python crawler.py --month 2026-03
```

### Q: 文件太大了怎么办？

每个月份文件通常在 100KB-500KB 左右（纯文本），取决于当月文章数量和长度。如果需要按年分目录，可以手动移动或修改代码中的 `OUTPUT_DIR` 逻辑。

### Q: 如何验证爬取是否完整？

```bash
# 1. 查看文章总数
python crawler.py --list-only 2>/dev/null | tail -1

# 2. 查看已爬取数量
cat .crawler_checkpoint.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['completed_urls']))"

# 3. 查看各月份文件的文章数
grep -c "^## " articles/*.md
```

## 开发

### 自定义扩展

如果需要修改爬虫行为，以下是关键函数：

| 函数 | 文件 | 说明 |
|------|------|------|
| `parse_article()` | crawler.py | 解析单篇文章的 HTML |
| `html_to_markdown()` | crawler.py | HTML → Markdown 转换逻辑 |
| `save_month_file()` | crawler.py | 月份文件的格式和保存逻辑 |
| `format_article_md()` | crawler.py | 单篇文章在 MD 文件中的格式 |

### 运行测试

```bash
# 测试爬取单个月份
python crawler.py --month 2026-03 --verbose

# 测试列表功能
python crawler.py --list-only
```

## License

MIT License. 本工具仅供学习和个人使用。
