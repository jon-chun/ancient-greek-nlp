#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetches articles from configured news sources (CNN or BBC), extracts text
using trafilatura, filters for keywords, saves text content in a specific
format with sequential/topic prefixes, ensures restartability, and generates
reports. Includes human-like browsing patterns, retry limits per source,
and skips video links. Aims for a target number of articles per source URL.
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Third-party imports - check if these are installed
try:
    import numpy as np
    import requests
    import trafilatura
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"CRITICAL ERROR: Missing required library. Please install dependencies. {e}", file=sys.stderr)
    sys.exit(1)


# --- Constants ---
KEYWORDS = ["america", "united states"]  # Case-insensitive check
TARGET_ARTICLES_PER_SOURCE_DEFAULT = 30 # Default target number of *new* articles per source URL
MAX_LINKS_PER_SOURCE_PAGE = 150
# --- ADDED RETRY LIMIT ---
MAX_RETRIES_PER_SOURCE = 50 # Max consecutive failures on a source URL before skipping it
REQUEST_TIMEOUT = 25
MIN_DELAY = 0.7
MAX_DELAY = 3.5
DELAY_POWER_PARAM = 5.0

# Realistic User-Agent Strings
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1'
]

# --- Source Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = SCRIPT_DIR.parent / 'data'

SOURCE_CONFIG = {
    "cnn": {
        "name": "CNN",
        "url_ls": [
            "https://www.cnn.com/us", "https://www.cnn.com/world", "https://www.cnn.com/business",
        ],
        "output_articles_dir": BASE_DATA_DIR / 'cnn_english' / 'article_download',
        "output_reports_dir": BASE_DATA_DIR / 'cnn_english' / 'analysis_reports',
        "base_domain": "cnn.com"
    },
    "bbc": {
        "name": "BBC",
        "url_ls": [
            'https://www.bbc.com/news', 'https://www.bbc.com/business', 'https://www.bbc.com/news/world',
            'https://www.bbc.com/news/uk', 'https://www.bbc.com/news/technology',
            'https://www.bbc.com/news/science_and_environment', 'https://www.bbc.com/news/health'
        ],
        "output_articles_dir": BASE_DATA_DIR / 'bbc_english' / 'article_download',
        "output_reports_dir": BASE_DATA_DIR / 'bbc_english' / 'analysis_reports',
        "base_domain": "bbc.com"
    }
}

# --- Logging Setup ---
LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NONE']
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set initial level high

def setup_logging(log_level_str: str):
    """Configures the logging level."""
    global logger
    if logger is None: logger = logging.getLogger(__name__)
    log_level = logging.DEBUG
    if log_level_str == 'NONE':
        log_level = logging.CRITICAL + 1
        logging.disable(logging.CRITICAL)
        print("Logging explicitly disabled.")
    else:
        try:
            log_level = getattr(logging, log_level_str.upper())
            logging.disable(logging.NOTSET)
        except AttributeError:
            print(f"Invalid log level: {log_level_str}. Defaulting to DEBUG.", file=sys.stderr)
            log_level = logging.DEBUG
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler): root_logger.removeHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)
        logger.debug("Added new StreamHandler to root logger.")
    else:
        logger.debug("StreamHandler already exists on root logger.")
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler): handler.setLevel(log_level); handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.propagate = True
    logger.info(f"Logging setup complete. Level set to {log_level_str.upper()}")

# --- Helper Functions ---

def scan_existing_files(directory: Path) -> dict:
    # ... (implementation unchanged from previous version with debug logs) ...
    logger.debug(f"Starting scan of directory: {directory}")
    existing_base_filenames = set()
    max_count = 0
    file_pattern = re.compile(r"^(\d+)_([a-zA-Z0-9_.-]+)_(.+)\.txt$")
    if not directory.is_dir():
        logger.warning(f"Output directory {directory} not found during scan. Will start fresh.")
        return {"bases": existing_base_filenames, "max_count": max_count}
    logger.info(f"Scanning existing files in: {directory}")
    files_scanned = 0
    for f in directory.glob("*.txt"):
        files_scanned += 1
        logger.debug(f"Scanning file: {f.name}")
        match = file_pattern.match(f.name)
        if match:
            try:
                count = int(match.group(1))
                base_name = match.group(3)
                existing_base_filenames.add(base_name) # Add the base part only
                if count > max_count: max_count = count
                logger.debug(f"Parsed existing file: {f.name} (Count: {count}, Base: '{base_name}')")
            except ValueError: logger.warning(f"Could not parse count from filename: {f.name}")
            except IndexError: logger.warning(f"Could not parse base filename from: {f.name}")
        else: logger.debug(f"Filename did not match pattern: {f.name}")
    logger.info(f"Scan complete. Scanned {files_scanned} *.txt files. Found {len(existing_base_filenames)} unique base filenames. Max existing count: {max_count}")
    if logger.isEnabledFor(logging.DEBUG):
        bases_to_log = list(existing_base_filenames)[:10]
        logger.debug(f"Existing base filenames sample (up to 10): {bases_to_log}")
    return {"bases": existing_base_filenames, "max_count": max_count}

def human_delay(min_sec: float = MIN_DELAY, max_sec: float = MAX_DELAY, power: float = DELAY_POWER_PARAM):
    # ... (implementation unchanged) ...
    delay_range = max_sec - min_sec
    if power <= 0: power = 1.0
    random_delay = (1 - np.random.power(power)) * delay_range + min_sec
    actual_delay = max(min_sec, min(random_delay, max_sec))
    logger.debug(f"Sleeping for {actual_delay:.2f} seconds...")
    time.sleep(actual_delay)
    logger.debug("Sleep finished.")

def sanitize_base_filename(url: str) -> str:
    # ... (implementation unchanged) ...
    try:
        parsed_url = urlparse(url)
        path_part = parsed_url.path.strip('/')
        path_part = re.sub(r'^\d{4}/\d{2}/\d{2}/', '', path_part)
        path_part = re.sub(r'-(\d+)$', '', path_part)
        path_part = re.sub(r'\.(html|htm|php|asp|aspx)$', '', path_part, flags=re.IGNORECASE)
        if not path_part and parsed_url.netloc: path_part = parsed_url.netloc
        elif not path_part: path_part = "unknown_article"
        safe_name = re.sub(r'[^\w\.\-]+', '_', path_part.replace('/', '_'))
        safe_name = re.sub(r'_+', '_', safe_name).strip('._')
        max_len = 100
        if len(safe_name) > max_len: safe_name = safe_name[:max_len] + "_etc"
        result = safe_name if safe_name else "sanitized_article"
        logger.debug(f"Sanitized URL '{url}' to base filename '{result}'")
        return result
    except Exception as e:
        logger.error(f"Error sanitizing URL {url} for base filename: {e}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"error_article_{timestamp}"

def get_topic_from_url(url: str, source: str) -> str:
    # ... (implementation unchanged) ...
    logger.debug(f"Extracting topic from URL: {url} for source: {source}")
    try:
        parsed_url = urlparse(url)
        path_segments = [seg for seg in parsed_url.path.strip('/').split('/') if seg]
        if not path_segments: logger.debug("No path segments found."); return "unknown"
        topic = "unknown"
        if source == "cnn":
            if len(path_segments) > 2 and all(s.isdigit() for s in path_segments[:3]):
                if len(path_segments) > 3: topic = path_segments[3]
            elif len(path_segments) > 1 and path_segments[0] in ['us', 'world', 'europe', 'asia', 'africa', 'middle-east', 'politics', 'business', 'health', 'entertainment', 'style', 'travel', 'sport', 'tech']:
                 if len(path_segments) > 1 and len(path_segments[1]) > 3 and not path_segments[1].isdigit(): topic = path_segments[1]
                 else: topic = path_segments[0]
            elif path_segments: topic = path_segments[0]
        elif source == "bbc":
            if len(path_segments) > 0:
                 potential_topic = path_segments[-1]
                 match = re.match(r'([a-zA-Z\-]+)-(\d+)$', potential_topic)
                 if match: topic = match.group(1)
                 elif len(path_segments) > 1:
                      if potential_topic.isdigit() and len(path_segments) > 1: topic = path_segments[-2]
                      else: topic = path_segments[0]
                 else: topic = path_segments[0]
        topic = topic.lower()
        topic = re.sub(r'[^\w\-]+', '_', topic)
        topic = re.sub(r'_+', '_', topic).strip('_')
        topic_result = topic if topic else "unknown"
        logger.debug(f"Extracted topic: {topic_result}")
        return topic_result
    except Exception as e:
        logger.error(f"Error extracting topic from URL {url} for source {source}: {e}")
        return "unknown"

def extract_title(html_content: str, url: str, source: str) -> str:
    # ... (implementation unchanged) ...
    logger.debug(f"Attempting to extract title from URL: {url}")
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        title = "Title Not Found"
        if source == "cnn":
            title_tag = soup.find('h1', {'data-editable': 'headlineText'})
            if title_tag and title_tag.get_text(strip=True): title = title_tag.get_text(strip=True)
        elif source == "bbc":
             title_tag = soup.find('h1', {'id': 'main-heading'})
             if title_tag and title_tag.get_text(strip=True): title = title_tag.get_text(strip=True)
        if title == "Title Not Found":
             for tag_name in ['h1', 'h2']:
                  tag = soup.find(tag_name)
                  if tag and tag.get_text(strip=True):
                       potential_title = tag.get_text(strip=True)
                       if len(potential_title) > 10: title = potential_title; break
             if title == "Title Not Found" and soup.title and soup.title.string:
                  potential_title = soup.title.string.strip()
                  if len(potential_title) > 10: title = potential_title
        if source == "cnn":
            title = re.sub(r'\s*\|\s*CNN.*$', '', title).strip()
            title = re.sub(r'^CNN\s*[:\-]?\s*', '', title).strip()
        elif source == "bbc":
             title = re.sub(r'\s*-\s*BBC News$', '', title).strip()
             title = re.sub(r'\s*-\s*BBC Worklife$', '', title).strip()
             title = re.sub(r'\s*-\s*BBC.*$', '', title).strip()
        if title == "Title Not Found": logger.warning(f"Could not extract title from: {url}")
        else: logger.debug(f"Extracted title: {title}")
        return title
    except Exception as e:
        logger.error(f"Error extracting title from {url}: {e}", exc_info=True)
        return "Title Extraction Error"

def extract_published_date(html_content: str, url: str) -> str:
    # ... (implementation unchanged) ...
    logger.debug(f"Attempting to extract published date from URL: {url}")
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        date_str = "Unknown"
        selectors = [
            'meta[property="article:published_time"]', 'meta[property="og:published_time"]',
            'meta[name="pubdate"]', 'meta[name="publishdate"]', 'meta[name="dc.date.issued"]',
            'meta[itemprop="datePublished"]', 'meta[name="date"]' ]
        for selector in selectors:
            tag = soup.select_one(selector)
            if tag and tag.get('content'): date_str = tag['content'].strip(); logger.debug(f"Found date via meta {selector}: {date_str}"); break
        if date_str == "Unknown":
            time_tag = soup.find('time', {'data-testid': 'timestamp'})
            if time_tag and time_tag.get('datetime'): date_str = time_tag['datetime'].strip(); logger.debug(f"Found date via BBC time tag: {date_str}")
            else:
                 time_tag = soup.find('time', datetime=True)
                 if time_tag and time_tag['datetime']: date_str = time_tag['datetime'].strip(); logger.debug(f"Found date via generic time tag: {date_str}")
        if date_str == "Unknown":
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    graph = data.get('@graph', [data]) if isinstance(data, dict) else [data]
                    for item in graph:
                         if isinstance(item, dict):
                              pub_date = item.get('datePublished') or item.get('uploadDate')
                              if pub_date: date_str = pub_date.strip(); logger.debug(f"Found date via JSON-LD: {date_str}"); break
                    if date_str != "Unknown": break
                except Exception as e: logger.debug(f"Error processing JSON-LD in {url}: {e}")
        if date_str != "Unknown":
            date_part = date_str.split('T')[0]
            date_part = re.sub(r'[/\._\s]+', '-', date_part)
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_part): logger.debug(f"Formatted date: {date_part}"); return date_part
            else: logger.debug(f"Returning raw date string: {date_str}"); return date_str
        logger.debug(f"Published date not found for: {url}")
        return "Unknown"
    except Exception as e:
        logger.error(f"Error extracting published date from {url}: {e}", exc_info=True)
        return "Unknown"

def generate_reports(successful: list, failed: list, report_dir: Path, source_name: str):
    # ... (implementation unchanged) ...
    logger.info(f"Attempting to generate reports for {source_name} in {report_dir}")
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_base_name = f"{source_name.lower()}_scraping_report"
        txt_report_path = report_dir / f"{report_base_name}.txt"
        json_report_path = report_dir / f"{report_base_name}.json"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        logger.info(f"Generating reports for {source_name} in: {report_dir}")
        successful.sort(key=lambda x: int(x['filename'].split('_')[0]) if x['filename'].split('_')[0].isdigit() else 0)
        failed.sort(key=lambda x: x['url'])
        with open(txt_report_path, 'w', encoding='utf-8') as f:
            f.write(f"{source_name} Scraping Report\nGenerated: {timestamp}\n{'='*30}\n\n")
            f.write(f"Successful Downloads ({len(successful)}):\n{'-'*25}\n")
            if successful:
                for item in successful: f.write(f"  File:  {item['filename']}\n  URL:   {item['url']}\n  Title: {item['title']}\n\n")
            else: f.write("  None\n\n")
            f.write(f"Failed Attempts ({len(failed)}):\n{'-'*20}\n")
            if failed:
                for item in failed:
                    details = f" ({item['details']})" if item.get('details') else ""
                    f.write(f"  URL:    {item['url']}\n  Reason: {item['reason']}{details}\n\n")
            else: f.write("  None\n\n")
        logger.info(f"TXT report generated: {txt_report_path}")
        report_data = {
            "source": source_name, "generation_timestamp": timestamp,
            "summary": {"successful_count": len(successful), "failed_count": len(failed)},
            "successful_downloads": successful, "failed_attempts": failed }
        with open(json_report_path, 'w', encoding='utf-8') as f: json.dump(report_data, f, indent=4)
        logger.info(f"JSON report generated: {json_report_path}")
    except Exception as e:
        logger.error(f"Failed to generate reports for {source_name}: {e}", exc_info=True)

def is_article_link(url: str, source_config: dict) -> bool:
    """Heuristic check if a URL likely points to an article page for the given source."""
    base_domain = source_config['base_domain']
    source = [k for k, v in SOURCE_CONFIG.items() if v == source_config][0]
    logger.debug(f"Checking if URL is article link for source {source}: {url}")
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']: logger.debug(f"[{url}] Skip: Invalid scheme ({parsed_url.scheme})"); return False
        netloc = parsed_url.netloc
        if not netloc or not (base_domain in netloc or netloc.endswith('.' + base_domain)): logger.debug(f"[{url}] Skip: Domain mismatch ({netloc} vs {base_domain})"); return False
        path = parsed_url.path.lower()
        excluded_extensions = ('.pdf', '.zip', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.xml', '.rss', '.css', '.js', '.ico', '.svg', '.txt', '.json', '. TIF', '.JPG')
        if path.endswith(excluded_extensions): logger.debug(f"[{url}] Skip: Excluded file extension ({path})"); return False

        # --- ADDED /video(s)/ to excluded prefixes ---
        excluded_prefixes = [
            '/videos/', '/video/', '/profiles/', '/specials/', '/live-news/', '/weather/', '/more/', '/shows/', '/vr/', '/collection/',
            '/podcasts/', '/audio/', '/gallery/', '/app/', '/email/', '/account/', '/search', '/terms', '/privacy',
            '/accessibility', '/contact', '/help', '/usingthebbc', '/aboutthebbc', '/sport/', '/travel/', '/style/',
            '/future/', '/culture/', '/worklife/', '/reel/', '/cnn-underscored/'
        ]
        if path == '/' or any(path.startswith(p) for p in excluded_prefixes): logger.debug(f"[{url}] Skip: Excluded path prefix ({path})"); return False

        path_segments = [seg for seg in path.strip('/') if seg] # Get non-empty segments
        logger.debug(f"[{url}] Path segments for source '{source}': {path_segments}")
        if source == "cnn":
            if len(path_segments) < 2: logger.debug(f"[{url}] Skip: CNN path too short (< 2 segments)"); return False
            if re.search(r'/\d{4}/\d{2}/\d{2}/', parsed_url.path): logger.debug(f"[{url}] Pass: CNN date pattern matched."); return True
            if len(path_segments) >= 3 and path_segments[-1]:
                if re.search(r'[a-z]', path_segments[-1]) and not path_segments[-1].isdigit(): logger.debug(f"[{url}] Pass: CNN path depth/slug matched."); return True
        elif source == "bbc":
            if re.search(r'/[a-z\-]+-\d+$', parsed_url.path): logger.debug(f"[{url}] Pass: BBC category-ID pattern matched."); return True
            # Check for /news/ Suffix without ID (live pages, etc.) - maybe exclude? Let's keep for now.
            if '/story/' in path or '/article/' in path: logger.debug(f"[{url}] Pass: BBC story/article path matched."); return True
            # Allow paths like /news/world or /news/technology if they weren't excluded above
            if len(path_segments) >= 2: logger.debug(f"[{url}] Pass: BBC path has >= 2 segments and not excluded."); return True

        logger.debug(f"[{url}] Skip: No matching article pattern found for {source.upper()}.")
        return False
    except Exception as e:
        logger.error(f"Error during is_article_link check for {url}: {e}", exc_info=True)
        return False


# --- Main Logic ---
def main():
    """Main execution function."""
    # --- SETUP LOGGING WITH DEBUG INITIALLY ---
    setup_logging('DEBUG') # Setup with DEBUG first to catch early issues
    logger.debug("Entering main function.")

    parser = argparse.ArgumentParser(description="Fetch and process News articles from multiple sources.")
    parser.add_argument('--source', choices=SOURCE_CONFIG.keys(), default='bbc', help='Select the news source to scrape.')
    # --- SET DEBUG AS DEFAULT LOG LEVEL ---
    parser.add_argument('--log-level', choices=LOG_LEVELS, default='DEBUG', help='Set the logging level.')
    parser.add_argument('--target-per-source', type=int, default=TARGET_ARTICLES_PER_SOURCE_DEFAULT, help='Target number of new articles to download per source URL.')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES_PER_SOURCE, help='Max consecutive failures on a source URL before skipping it.')


    try: args = parser.parse_args(); logger.debug(f"Arguments parsed: {args}")
    except Exception as e: logger.critical(f"Error parsing command line arguments: {e}", exc_info=True); sys.exit(1)

    try: config = SOURCE_CONFIG[args.source]; logger.info(f"Using configuration for source: {args.source.upper()}")
    except KeyError: logger.critical(f"Invalid source '{args.source}'. Available: {list(SOURCE_CONFIG.keys())}"); sys.exit(1)

    current_target_per_source = args.target_per_source
    current_max_retries = args.max_retries # Use value from args
    source_name = config["name"]
    output_articles_dir = config["output_articles_dir"]
    output_reports_dir = config["output_reports_dir"]
    url_search_ls = config["url_ls"]

    # Re-setup logging with the level from arguments
    setup_logging(args.log_level)

    logger.info(f"--- Starting {source_name} Article Scraper ---")
    logger.info(f"Source: {args.source.upper()}")
    logger.info(f"Target Articles per Source URL: {current_target_per_source}")
    logger.info(f"Max Consecutive Retries per Source URL: {current_max_retries}") # Log max retries
    logger.info(f"Article Output Dir: {output_articles_dir}")
    logger.info(f"Reports Output Dir: {output_reports_dir}")
    logger.debug(f"Full Source URL List: {url_search_ls}")
    logger.debug(f"Keywords: {KEYWORDS}")

    logger.info("Ensuring output directories exist...")
    try:
        output_articles_dir.mkdir(parents=True, exist_ok=True); logger.debug(f"Checked/Created articles dir: {output_articles_dir}")
        output_reports_dir.mkdir(parents=True, exist_ok=True); logger.debug(f"Checked/Created reports dir: {output_reports_dir}")
        logger.info("Output directories ready.")
    except OSError as e: logger.critical(f"Could not create output directories: {e}", exc_info=True); sys.exit(1)

    logger.info("Starting scan for existing files...")
    existing_files_data = scan_existing_files(output_articles_dir)
    existing_base_filenames = existing_files_data["bases"]
    overall_download_count = existing_files_data["max_count"]
    logger.info(f"Initialization complete. Starting download count at: {overall_download_count + 1}")

    successful_scrapes = []; failed_scrapes = []
    logger.debug("Initializing requests session."); session = requests.Session()
    logger.debug("Compiling keyword regex."); keyword_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in KEYWORDS) + r')\b', re.IGNORECASE)

    logger.info("Starting main loop over source URLs.")
    total_downloaded_this_run = 0
    for start_url in url_search_ls:
        logger.info(f"===== Processing Source URL: {start_url} =====")
        source_new_downloads = 0; processed_links_from_source = 0
        consecutive_failures = 0 # Initialize retry counter for this source URL

        session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        source_page_html = None
        logger.info(f"Fetching source page: {start_url}")
        try:
            response = session.get(start_url, timeout=REQUEST_TIMEOUT); logger.debug(f"Source page fetch status code: {response.status_code}")
            response.raise_for_status(); source_page_html = response.text
            logger.info(f"-> Fetched source page successfully.")
        except requests.exceptions.RequestException as e: logger.error(f"-> FAILED to fetch source page {start_url}: {e}", exc_info=False); failed_scrapes.append({"url": start_url, "reason": "Source Page Fetch Error", "details": str(e)}); logger.warning(f"Skipping this source URL."); continue
        except Exception as e: logger.critical(f"-> UNEXPECTED error fetching source page {start_url}: {e}", exc_info=True); failed_scrapes.append({"url": start_url, "reason": "Source Page Fetch Error", "details": str(e)}); logger.warning(f"Skipping this source URL."); continue

        logger.info("Parsing source page and extracting links...")
        soup = BeautifulSoup(source_page_html, 'lxml')
        links_on_page = set(); total_a_tags = 0
        for a_tag in soup.find_all('a', href=True):
            total_a_tags += 1; href = a_tag['href']
            logger.debug(f"Processing found link: {href}")
            try:
                absolute_url = urljoin(start_url, href); parsed = urlparse(absolute_url)
                absolute_url = parsed._replace(fragment="").geturl()
                if is_article_link(absolute_url, config):
                    if absolute_url not in links_on_page: logger.debug(f"Adding valid article link: {absolute_url}"); links_on_page.add(absolute_url)
                    else: logger.debug(f"Duplicate article link found: {absolute_url}")
            except Exception as e: logger.warning(f"Could not process link '{href}' from {start_url}: {e}")

        logger.debug(f"Found {total_a_tags} total <a> tags on source page.")
        if not links_on_page: logger.warning(f"No potential article links found after filtering on source page: {start_url}"); continue
        logger.info(f"Found {len(links_on_page)} unique potential article links. Processing up to {MAX_LINKS_PER_SOURCE_PAGE}.")
        article_list = list(links_on_page); random.shuffle(article_list)
        logger.debug(f"Shuffled {len(article_list)} links for processing.")

        logger.info(f"Starting processing for {len(article_list)} links from {start_url}...")
        for i, article_url in enumerate(article_list):
            logger.debug(f"--- Iteration {i+1}/{len(article_list)} for source {start_url} ---")

            # --- Check overall link processing limit ---
            if processed_links_from_source >= MAX_LINKS_PER_SOURCE_PAGE: logger.info(f"Reached link limit ({MAX_LINKS_PER_SOURCE_PAGE}) for source: {start_url}"); break
            processed_links_from_source += 1

            # --- Check target downloads for this source ---
            if source_new_downloads >= current_target_per_source: logger.info(f"Target ({current_target_per_source}) met for source URL {start_url} in this run. Skipping remaining links."); break

            # --- Check consecutive failure limit ---
            if consecutive_failures >= current_max_retries:
                logger.warning(f"Reached max consecutive failures ({current_max_retries}) for source {start_url}. Skipping remaining links for this source.")
                break # Stop processing this source URL

            logger.info(f"Attempting article [{processed_links_from_source}/{MAX_LINKS_PER_SOURCE_PAGE} | Failures: {consecutive_failures}/{current_max_retries}]: {article_url}")

            # --- Assume failure until success ---
            current_attempt_failed = True

            logger.debug("Sanitizing URL for base filename check...")
            base_filename = sanitize_base_filename(article_url)
            logger.debug(f"Checking existence for base filename: '{base_filename}'")
            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Current existing base filenames set (sample): {list(existing_base_filenames)[:20]}")
            if base_filename in existing_base_filenames:
                logger.info(f"-> Skipping: Already downloaded (base name '{base_filename}')")
                # NOTE: Skipping an existing file does *not* count as a failure for the retry counter
                current_attempt_failed = False # Not a failure in terms of retrying
                continue # Go to next article URL
            else: logger.debug("Base filename not found in existing set.")

            logger.debug("Applying standard human delay...")
            human_delay()

            article_html, fetch_error = None, None
            logger.debug(f"Attempting to fetch article content...")
            try:
                current_user_agent = random.choice(USER_AGENTS)
                session.headers.update({'User-Agent': current_user_agent})
                logger.debug(f"Fetching with User-Agent: {current_user_agent}")
                response = session.get(article_url, timeout=REQUEST_TIMEOUT);
                logger.debug(f"Article fetch status code: {response.status_code}")
                response.raise_for_status()
                article_html = response.text
                logger.debug(f"-> Fetch successful.")
            except requests.exceptions.HTTPError as e: fetch_error = f"HTTP Error {e.response.status_code} {e.response.reason}"; logger.warning(f"-> Fetch Error: {fetch_error}"); logger.debug("Applying longer human delay after HTTP error..."); human_delay(min_sec=3.0, max_sec=10.0)
            except requests.exceptions.Timeout: fetch_error = "Timeout"; logger.warning(f"-> Fetch Error: {fetch_error}")
            except requests.exceptions.RequestException as e: fetch_error = f"Request Error: {e}"; logger.warning(f"-> Fetch Error: {fetch_error}")
            except Exception as e: fetch_error = f"Unexpected Fetch Error: {e}"; logger.error(f"-> Fetch Error: {fetch_error}", exc_info=True)

            if fetch_error:
                failed_scrapes.append({"url": article_url, "reason": "Fetch Error", "details": fetch_error})
                logger.debug(f"Continuing to next article after fetch error.")
                consecutive_failures += 1 # Increment failure counter
                continue
            if article_html is None:
                logger.error(f"HTML None after fetch: {article_url}")
                failed_scrapes.append({"url": article_url, "reason": "Fetch Error", "details": "HTML content was None after fetch attempt"})
                logger.debug(f"Continuing to next article after None HTML.")
                consecutive_failures += 1 # Increment failure counter
                continue
            logger.debug(f"Article HTML content length: {len(article_html)}")

            logger.debug("Extracting metadata and text...")
            title = extract_title(article_html, article_url, args.source)
            if title == "Title Extraction Error":
                logger.warning(f"-> Title extraction failed.")
                failed_scrapes.append({"url": article_url, "reason": "Title Error", "details": "Extraction function failed"}); logger.debug(f"Continuing after title error.");
                consecutive_failures += 1 # Increment failure counter
                continue
            published_date = extract_published_date(article_html, article_url)
            article_text, extraction_error = None, None
            logger.debug("Attempting text extraction with Trafilatura...")
            try: article_text = trafilatura.extract(article_html, include_comments=False, include_tables=False, favor_precision=True)
            except Exception as e: extraction_error = f"Trafilatura Error: {e}"; logger.error(f"{extraction_error} extracting text from {article_url}", exc_info=True)

            if extraction_error:
                logger.warning(f"-> Text extraction failed: {extraction_error}")
                failed_scrapes.append({"url": article_url, "reason": "Extraction Error", "details": extraction_error}); logger.debug(f"Continuing after extraction error.");
                consecutive_failures += 1 # Increment failure counter
                continue
            if not article_text:
                logger.warning(f"-> Empty text extracted.")
                failed_scrapes.append({"url": article_url, "reason": "Extraction Error", "details": "Trafilatura returned empty text"}); logger.debug(f"Continuing after empty text.");
                consecutive_failures += 1 # Increment failure counter
                continue
            logger.debug(f"Successfully extracted text (Length: {len(article_text)})")
            logger.debug(f"Extracted Title: '{title}', Published Date: '{published_date}'")

            save_article = False
            logger.debug("Checking for keywords...")
            try:
                if keyword_pattern.search(article_text): logger.debug(f"Keywords FOUND."); save_article = True
                else:
                    logger.info(f"-> Skipping: Keywords NOT found.")
                    failed_scrapes.append({"url": article_url, "reason": "Keywords Not Found", "details": ""})
                    consecutive_failures += 1 # Increment failure counter
            except Exception as e:
                logger.error(f"Keyword check error for {article_url}: {e}", exc_info=True);
                failed_scrapes.append({"url": article_url, "reason": "Keyword Check Error", "details": str(e)}); logger.debug(f"Continuing after keyword check error.");
                consecutive_failures += 1 # Increment failure counter
                continue

            if save_article:
                logger.debug("Article contains keywords. Preparing to save...")
                overall_download_count += 1; source_new_downloads += 1; total_downloaded_this_run += 1
                topic = get_topic_from_url(article_url, args.source)
                # base_filename already generated for existence check
                full_filename = f"{overall_download_count}_{topic}_{base_filename}.txt"
                filepath = output_articles_dir / full_filename
                scraped_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.debug(f"Generated filename: {full_filename}")
                logger.debug(f"Saving to path: {filepath}")
                file_content = (f"###FORMAT:\nTitle: {title}\nURL: {article_url}\n"
                                f"Scraped Date: {scraped_timestamp}\nPublished Date: {published_date}\n\n"
                                f"--- Text ---\n{article_text}\n")
                logger.debug(f"File content length: {len(file_content)}")
                try:
                    logger.debug(f"Attempting to write file: {filepath}...")
                    with open(filepath, 'w', encoding='utf-8') as f: f.write(file_content)
                    logger.info(f"-> SUCCESS: Saved '{filepath.name}' (Source: {source_new_downloads}/{current_target_per_source}, Overall: {overall_download_count})")
                    successful_scrapes.append({"url": article_url, "title": title, "filename": filepath.name})
                    existing_base_filenames.add(base_filename); logger.debug(f"Added '{base_filename}' to existing base filenames set.")
                    # --- Reset failure counter on success ---
                    consecutive_failures = 0
                    current_attempt_failed = False # Mark as success
                except Exception as e:
                    logger.error(f"-> SAVE FAILED for {filepath}: {e}", exc_info=True)
                    failed_scrapes.append({"url": article_url, "reason": "Save Error", "details": str(e)})
                    overall_download_count -= 1; source_new_downloads -= 1; total_downloaded_this_run -=1
                    logger.warning("Rolled back download counts due to save error.")
                    consecutive_failures += 1 # Increment failure counter
                    # current_attempt_failed remains True
            else:
                 # If save_article was False (due to keyword miss), failure counter was already incremented
                 pass

            # This check is redundant now as failure increments happen within the specific fail blocks
            # if current_attempt_failed:
            #     consecutive_failures += 1
            #     logger.debug(f"Incremented consecutive failure count to {consecutive_failures}")

            logger.debug(f"Finished processing iteration for {article_url}")
        # End of article loop for one source URL
        logger.info(f"Finished processing source URL: {start_url}. Articles saved in this run for this source: {source_new_downloads}")

    # --- Generate Final Reports ---
    logger.info("Finished processing all source URLs. Generating final reports...")
    generate_reports(successful_scrapes, failed_scrapes, output_reports_dir, source_name)

    logger.info(f"===== {source_name} Scraping Run Complete =====")
    logger.info(f"Total articles saved in this run: {total_downloaded_this_run}")
    logger.info(f"Total failed attempts recorded: {len(failed_scrapes)}")
    logger.info(f"Final overall download count (based on filenames): {overall_download_count}")
    logger.info(f"Article files saved in: {output_articles_dir}")
    logger.info(f"Reports saved in: {output_reports_dir}")
    logger.info("Script finished.")


if __name__ == "__main__":
    print("Script execution started...") # Basic print before logging setup
    try: main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.", file=sys.stderr)
        try:
             if logger and logger.handlers: logger.warning("Process interrupted by user.")
        except NameError: pass
        sys.exit(0)
    except Exception as e:
        try:
            if logger and logger.handlers: logger.critical(f"An unexpected critical error occurred in main execution: {e}", exc_info=True)
            else: print(f"CRITICAL ERROR: {e}", file=sys.stderr); import traceback; traceback.print_exc()
        except NameError: print(f"CRITICAL ERROR (logging unavailable): {e}", file=sys.stderr); import traceback; traceback.print_exc()
        sys.exit(1)
    finally: print("Script execution finished.") # Always print this

