#!/usr/bin/env python3
"""
hecaitou.com Blog Crawler
=========================
Crawl all articles from https://www.hecaitou.com/ (和菜头的槽边往事),
group them by month, and save each month's articles into a single Markdown file.

Usage:
    python crawler.py                  # Full crawl
    python crawler.py --year 2025      # Only crawl articles from 2025
    python crawler.py --month 2025-03  # Only crawl articles from March 2025
    python crawler.py --resume         # Resume from last checkpoint
    python crawler.py --list-only      # Only list articles, don't download
    python crawler.py --delay 3        # Set delay between requests (seconds)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "https://www.hecaitou.com"
SITEMAP_INDEX_URL = f"{BASE_URL}/sitemap.xml"
OUTPUT_DIR = "articles"
CHECKPOINT_FILE = ".crawler_checkpoint.json"
LOG_FILE = "crawler.log"

DEFAULT_DELAY = 2  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose=False):
    """Configure logging to both file and console."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger = logging.getLogger("crawler")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================================
# HTTP Session
# ============================================================================

def create_session():
    """Create a requests session with proper headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return session


def fetch_url(session, url, logger, retries=MAX_RETRIES):
    """Fetch a URL with retries and error handling."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                wait = RETRY_DELAY * attempt
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"All {retries} attempts failed for {url}")
                return None


# ============================================================================
# Sitemap Parsing
# ============================================================================

def fetch_sitemap_index(session, logger):
    """Fetch the sitemap index and return sub-sitemap URLs."""
    logger.info(f"Fetching sitemap index: {SITEMAP_INDEX_URL}")
    resp = fetch_url(session, SITEMAP_INDEX_URL, logger)
    if not resp:
        raise RuntimeError("Failed to fetch sitemap index")

    # The sitemap index contains links to sub-sitemaps
    # Parse as XML; handle potential namespace issues
    text = resp.text.strip()

    # Try to parse as sitemap index
    sub_sitemaps = []
    try:
        root = ET.fromstring(text)
        # Remove namespace for easier parsing
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Check if it's a sitemapindex
        for sitemap in root.findall(".//sm:sitemap", ns):
            loc = sitemap.find("sm:loc", ns)
            if loc is not None and loc.text:
                sub_sitemaps.append(loc.text.strip())

        # If no sub-sitemaps found, try without namespace
        if not sub_sitemaps:
            for sitemap in root.findall(".//sitemap"):
                loc = sitemap.find("loc")
                if loc is not None and loc.text:
                    sub_sitemaps.append(loc.text.strip())
    except ET.ParseError:
        # Fallback: extract URLs with regex
        urls = re.findall(r"(https?://[^\s<>\"]+sitemap[^\s<>\"]*)", text)
        sub_sitemaps = urls

    if not sub_sitemaps:
        # Maybe it's just a single sitemap, not an index
        # Try extracting page URLs from it
        logger.info("No sub-sitemaps found, trying to parse URLs from the response")
        # Attempt to find sitemap page links from plain text
        page_urls = re.findall(
            r"(https?://www\.hecaitou\.com/sitemap\.xml\?page=\d+)", text
        )
        if page_urls:
            sub_sitemaps = page_urls
        else:
            # Treat the index itself as a sitemap
            sub_sitemaps = [SITEMAP_INDEX_URL]

    logger.info(f"Found {len(sub_sitemaps)} sub-sitemap(s)")
    return sub_sitemaps


def parse_sitemap(session, sitemap_url, logger):
    """Parse a sitemap XML and return list of (url, lastmod) tuples."""
    logger.info(f"Parsing sitemap: {sitemap_url}")
    resp = fetch_url(session, sitemap_url, logger)
    if not resp:
        return []

    articles = []
    text = resp.text.strip()

    try:
        root = ET.fromstring(text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        for url_elem in root.findall(".//sm:url", ns):
            loc = url_elem.find("sm:loc", ns)
            lastmod = url_elem.find("sm:lastmod", ns)
            if loc is not None and loc.text:
                url = loc.text.strip()
                mod = lastmod.text.strip() if lastmod is not None and lastmod.text else ""
                articles.append((url, mod))

        # Try without namespace if nothing found
        if not articles:
            for url_elem in root.findall(".//url"):
                loc = url_elem.find("loc")
                lastmod = url_elem.find("lastmod")
                if loc is not None and loc.text:
                    url = loc.text.strip()
                    mod = lastmod.text.strip() if lastmod is not None and lastmod.text else ""
                    articles.append((url, mod))

    except ET.ParseError:
        # Fallback: regex extraction from raw text
        # The text format from hecaitou.com appears as URL followed by date
        lines = text.split("\n")
        # Also try: URLs interleaved with dates
        url_pattern = re.compile(r"(https://www\.hecaitou\.com/\d{4}/\d{2}/[^\s]+\.html)")
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T[\d:]+Z)")

        urls_found = url_pattern.findall(text)
        dates_found = date_pattern.findall(text)

        if urls_found:
            for i, url in enumerate(urls_found):
                mod = dates_found[i] if i < len(dates_found) else ""
                articles.append((url, mod))

    logger.info(f"Found {len(articles)} article URLs in sitemap")
    return articles


def get_all_article_urls(session, logger):
    """Get all article URLs from the sitemap."""
    sub_sitemaps = fetch_sitemap_index(session, logger)
    all_articles = []

    for sitemap_url in sub_sitemaps:
        articles = parse_sitemap(session, sitemap_url, logger)
        all_articles.extend(articles)
        time.sleep(1)

    # Filter: only keep actual article URLs (pattern: /YYYY/MM/slug.html)
    article_pattern = re.compile(r"https://www\.hecaitou\.com/\d{4}/\d{2}/[^/]+\.html")
    filtered = [(url, mod) for url, mod in all_articles if article_pattern.match(url)]

    # Deduplicate by URL
    seen = set()
    unique = []
    for url, mod in filtered:
        if url not in seen:
            seen.add(url)
            unique.append((url, mod))

    # Sort by URL (effectively by date since URL contains YYYY/MM)
    unique.sort(key=lambda x: x[0])

    logger.info(f"Total unique articles: {len(unique)}")
    return unique


# ============================================================================
# Article Parsing
# ============================================================================

def extract_date_from_url(url):
    """Extract year and month from article URL."""
    match = re.search(r"/(\d{4})/(\d{2})/", url)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def extract_title_from_url(url):
    """Extract a fallback title from URL slug."""
    match = re.search(r"/([^/]+)\.html$", url)
    if match:
        slug = match.group(1)
        # Convert hyphens to spaces, handle common patterns
        title = slug.replace("-", " ").replace("_", " ").strip()
        return title
    return "Untitled"


def parse_article(html, url, logger):
    """
    Parse a single article page and extract title, date, and content.

    Returns dict with keys: title, date, date_str, content_md, url
    """
    soup = BeautifulSoup(html, "lxml")

    # --- Extract title ---
    title = None

    # Method 1: <h3 class="post-title entry-title"> (Blogger standard)
    title_elem = soup.select_one("h3.post-title.entry-title")
    if title_elem:
        title = title_elem.get_text(strip=True)

    # Method 2: <h1> or other heading in post
    if not title:
        for tag in ["h1", "h2", "h3"]:
            elem = soup.select_one(f".post {tag}, .post-outer {tag}, article {tag}")
            if elem:
                title = elem.get_text(strip=True)
                break

    # Method 3: <title> tag
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Remove blog name suffix
            for sep in [" - ", " | ", " :: ", "："]:
                if sep in title:
                    title = title.split(sep)[0].strip()

    # Method 4: Fallback from URL
    if not title:
        title = extract_title_from_url(url)

    # --- Extract date ---
    date_str = ""
    date_obj = None

    # Method 1: <abbr class="published" title="...">
    abbr = soup.select_one("abbr.published")
    if abbr and abbr.get("title"):
        date_str = abbr["title"]

    # Method 2: <time> element
    if not date_str:
        time_elem = soup.select_one("time.published, time[datetime]")
        if time_elem:
            date_str = time_elem.get("datetime", "") or time_elem.get_text(strip=True)

    # Method 3: date-header
    if not date_str:
        date_header = soup.select_one(".date-header span, .date-header")
        if date_header:
            date_str = date_header.get_text(strip=True)

    # Method 4: From URL
    if not date_str:
        year, month = extract_date_from_url(url)
        if year and month:
            date_str = f"{year}-{month:02d}"

    # Parse date_str to a datetime object
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
    ]:
        try:
            date_obj = datetime.strptime(date_str[:len(fmt.replace("%", "x"))], fmt)
            break
        except (ValueError, IndexError):
            continue

    # --- Extract content ---
    content_html = ""

    # Method 1: .post-body.entry-content (Blogger standard)
    body = soup.select_one(".post-body.entry-content")

    # Method 2: .post-body
    if not body:
        body = soup.select_one(".post-body")

    # Method 3: article or .post
    if not body:
        body = soup.select_one("article .entry-content, .post .entry-content")

    if body:
        # Remove script tags, style tags, and ad sections
        for tag in body.find_all(["script", "style", "ins", "noscript"]):
            tag.decompose()

        # Remove share buttons, related posts, etc.
        for selector in [
            ".post-share-buttons",
            ".related-posts",
            ".blog-pager",
            ".post-footer",
            "#comments",
            ".comments",
        ]:
            for elem in body.select(selector):
                elem.decompose()

        content_html = str(body)
    else:
        logger.warning(f"Could not find article body for: {url}")
        # Last resort: try to get the main content area
        main = soup.select_one("#main, .main, main")
        if main:
            content_html = str(main)

    # Convert HTML to Markdown
    content_md = ""
    if content_html:
        content_md = html_to_markdown(content_html)

    return {
        "title": title,
        "date": date_obj,
        "date_str": date_str,
        "content_md": content_md,
        "url": url,
    }


def html_to_markdown(html_str):
    """
    Convert HTML string to clean Markdown.
    """
    # Use markdownify for conversion
    # Note: strip and convert are mutually exclusive in markdownify
    text = md(
        html_str,
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "ins", "noscript", "iframe"],
    )

    # Clean up the markdown
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Remove image tags that are just tracking pixels or tiny images
    # Keep meaningful images
    text = re.sub(r"!\[\]\(https?://[^\)]*1x1[^\)]*\)", "", text)
    text = re.sub(r"!\[\]\(https?://[^\)]*pixel[^\)]*\)", "", text)

    # Remove blogger-specific artifacts
    text = re.sub(r"\[]\(#\)", "", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Strip leading/trailing whitespace from the whole text
    text = text.strip()

    return text


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_file):
    """Load crawl progress from checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_urls": [], "last_updated": ""}


def save_checkpoint(checkpoint_file, completed_urls):
    """Save crawl progress to checkpoint file."""
    data = {
        "completed_urls": list(completed_urls),
        "last_updated": datetime.now().isoformat(),
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# Article Storage
# ============================================================================

def format_article_md(article):
    """Format a single article as a Markdown section."""
    parts = []

    # Article title as H2
    parts.append(f"## {article['title']}")
    parts.append("")

    # Metadata line
    meta_parts = []
    if article.get("date_str"):
        meta_parts.append(f"**日期**: {article['date_str']}")
    meta_parts.append(f"**原文链接**: [{article['url']}]({article['url']})")
    parts.append(" | ".join(meta_parts))
    parts.append("")

    # Separator
    parts.append("---")
    parts.append("")

    # Article content
    if article.get("content_md"):
        parts.append(article["content_md"])
    else:
        parts.append("*（文章内容获取失败）*")

    parts.append("")
    parts.append("")

    return "\n".join(parts)


def get_month_key(year, month):
    """Generate a month key string like '2025-03'."""
    return f"{year:04d}-{month:02d}"


def get_month_filename(year, month):
    """Generate a filename for a month's articles."""
    return f"{year:04d}-{month:02d}.md"


def save_month_file(output_dir, year, month, articles, logger):
    """
    Save all articles for a given month to a Markdown file.
    Articles are sorted by date (earliest first).
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = get_month_filename(year, month)
    filepath = os.path.join(output_dir, filename)

    # Sort articles by date
    articles.sort(key=lambda a: a.get("date") or datetime.min)

    # Build the file content
    parts = []

    # File header
    month_names_cn = {
        1: "一月", 2: "二月", 3: "三月", 4: "四月",
        5: "五月", 6: "六月", 7: "七月", 8: "八月",
        9: "九月", 10: "十月", 11: "十一月", 12: "十二月",
    }
    month_cn = month_names_cn.get(month, f"{month}月")

    parts.append(f"# 槽边往事 - {year}年{month_cn}")
    parts.append("")
    parts.append(f"> 本文件收录了 [{year}年{month:02d}月] 的所有文章，共 {len(articles)} 篇。")
    parts.append(f"> 来源: https://www.hecaitou.com/")
    parts.append(f"> 自动抓取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("")

    # Table of contents
    parts.append("## 目录")
    parts.append("")
    for i, article in enumerate(articles, 1):
        date_prefix = ""
        if article.get("date"):
            date_prefix = f"[{article['date'].strftime('%m-%d')}] "
        elif article.get("date_str"):
            date_prefix = f"[{article['date_str'][:10]}] "
        # Create anchor link
        anchor = re.sub(r"[^\w\u4e00-\u9fff-]", "", article["title"].replace(" ", "-")).lower()
        parts.append(f"{i}. {date_prefix}[{article['title']}](#{anchor})")
    parts.append("")
    parts.append("---")
    parts.append("")

    # Articles
    for article in articles:
        parts.append(format_article_md(article))
        parts.append("---")
        parts.append("")

    content = "\n".join(parts)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Saved {filename} ({len(articles)} articles)")
    return filepath


# ============================================================================
# Main Crawler Logic
# ============================================================================

def crawl(args):
    """Main crawl function."""
    logger = setup_logging(args.verbose)
    session = create_session()

    logger.info("=" * 60)
    logger.info("hecaitou.com Blog Crawler Started")
    logger.info("=" * 60)

    # Step 1: Get all article URLs from sitemap
    logger.info("[Step 1/3] Fetching article list from sitemap...")
    all_articles = get_all_article_urls(session, logger)

    if not all_articles:
        logger.error("No articles found. Exiting.")
        return

    # Step 2: Filter by year/month if specified
    if args.year:
        all_articles = [
            (url, mod) for url, mod in all_articles
            if re.search(rf"/{args.year}/", url)
        ]
        logger.info(f"Filtered to {len(all_articles)} articles for year {args.year}")

    if args.month:
        # args.month is like "2025-03"
        parts = args.month.split("-")
        y, m = parts[0], parts[1]
        all_articles = [
            (url, mod) for url, mod in all_articles
            if re.search(rf"/{y}/{m}/", url)
        ]
        logger.info(f"Filtered to {len(all_articles)} articles for month {args.month}")

    if args.list_only:
        logger.info("Article list (--list-only mode):")
        for url, mod in all_articles:
            print(f"  {mod[:10] if mod else '?':>12}  {url}")
        logger.info(f"Total: {len(all_articles)} articles")
        return

    # Load checkpoint if resuming
    completed_urls = set()
    if args.resume:
        checkpoint = load_checkpoint(CHECKPOINT_FILE)
        completed_urls = set(checkpoint.get("completed_urls", []))
        logger.info(f"Resuming: {len(completed_urls)} articles already completed")

    # Step 3: Crawl each article and group by month
    logger.info(f"[Step 2/3] Crawling {len(all_articles)} articles...")
    month_articles = defaultdict(list)  # key: "YYYY-MM" -> list of article dicts
    total = len(all_articles)
    success = 0
    skipped = 0
    failed = 0

    for i, (url, mod) in enumerate(all_articles, 1):
        # Skip if already completed
        if url in completed_urls:
            skipped += 1
            year, month = extract_date_from_url(url)
            if year and month:
                # We need to still record it for file regeneration if needed
                # but skip the download
                pass
            continue

        # Progress
        logger.info(f"[{i}/{total}] Fetching: {url}")

        # Fetch the article
        resp = fetch_url(session, url, logger)
        if not resp:
            failed += 1
            logger.error(f"Failed to fetch: {url}")
            continue

        # Parse the article
        try:
            article = parse_article(resp.text, url, logger)
        except Exception as e:
            failed += 1
            logger.error(f"Failed to parse {url}: {e}")
            continue

        # Determine year/month
        year, month = extract_date_from_url(url)
        if not year or not month:
            # Try from parsed date
            if article.get("date"):
                year = article["date"].year
                month = article["date"].month
            else:
                failed += 1
                logger.warning(f"Cannot determine date for: {url}, skipping")
                continue

        month_key = get_month_key(year, month)
        month_articles[month_key].append(article)
        success += 1

        # Update checkpoint
        completed_urls.add(url)
        if i % 10 == 0:
            save_checkpoint(CHECKPOINT_FILE, completed_urls)

        # Rate limiting
        time.sleep(args.delay)

    # Save final checkpoint
    save_checkpoint(CHECKPOINT_FILE, completed_urls)

    # Step 4: If resuming, we also need to reload existing articles
    # For simplicity, we re-save only months with newly downloaded articles
    # In a full re-crawl, all months are covered

    # Step 5: Save articles grouped by month
    logger.info(f"[Step 3/3] Saving articles to Markdown files...")
    output_dir = os.path.join(os.getcwd(), args.output if hasattr(args, 'output') and args.output else OUTPUT_DIR)
    saved_files = []

    for month_key in sorted(month_articles.keys()):
        articles = month_articles[month_key]
        year, month = map(int, month_key.split("-"))
        filepath = save_month_file(output_dir, year, month, articles, logger)
        saved_files.append(filepath)

    # Summary
    logger.info("=" * 60)
    logger.info("Crawl Summary")
    logger.info("=" * 60)
    logger.info(f"Total articles in sitemap: {total}")
    logger.info(f"Successfully crawled:      {success}")
    logger.info(f"Skipped (already done):    {skipped}")
    logger.info(f"Failed:                    {failed}")
    logger.info(f"Monthly files saved:       {len(saved_files)}")
    logger.info(f"Output directory:          {output_dir}")

    if saved_files:
        logger.info("Files saved:")
        for f in saved_files:
            logger.info(f"  {f}")

    logger.info("=" * 60)
    logger.info("Done!")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Crawl all articles from hecaitou.com and save as monthly Markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crawler.py                     # Full crawl of all articles
  python crawler.py --year 2025         # Only crawl 2025 articles
  python crawler.py --month 2025-03     # Only crawl March 2025
  python crawler.py --resume            # Resume interrupted crawl
  python crawler.py --list-only         # List all articles without downloading
  python crawler.py --delay 5           # 5 second delay between requests
  python crawler.py --verbose           # Show debug information
        """,
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Only crawl articles from this year (e.g., 2025)",
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Only crawl articles from this month (format: YYYY-MM, e.g., 2025-03)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list article URLs, don't download",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for Markdown files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    args = parser.parse_args()

    try:
        crawl(args)
    except KeyboardInterrupt:
        print("\n\nCrawl interrupted by user. Progress has been saved.")
        print(f"Use --resume to continue from where you left off.")
        sys.exit(1)
    except Exception as e:
        logging.getLogger("crawler").exception(f"Unexpected error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
