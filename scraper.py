"""
Scrapes SHL product catalog (Individual Test Solutions only).
Run once: python scraper.py
Outputs: catalog.json
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_BASE = "https://www.shl.com/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def scrape_listing_page(start: int) -> list[dict]:
    """Extract Individual Test Solutions rows from one catalog page."""
    url = f"{CATALOG_BASE}?start={start}&type=1"
    resp = requests.get(url, headers=HEADERS, allow_redirects=True, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")

    items = []
    # Find the table whose first th contains "Individual Test Solutions"
    for table in soup.find_all("table"):
        header = table.find("th")
        if header and "Individual Test Solutions" in header.get_text():
            for row in table.find_all("tr")[1:]:  # skip header row
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue

                link = cols[0].find("a")
                if not link:
                    continue

                name = link.get_text(strip=True)
                relative_url = link.get("href", "")
                # Normalise to full URL
                if relative_url.startswith("/"):
                    full_url = "https://www.shl.com" + relative_url
                else:
                    full_url = relative_url

                remote_testing = bool(cols[1].find("span", class_="-yes"))
                adaptive = bool(cols[2].find("span", class_="-yes"))
                test_types = [
                    s.get_text(strip=True)
                    for s in cols[3].find_all("span", class_="product-catalogue__key")
                ]

                items.append(
                    {
                        "name": name,
                        "url": full_url,
                        "remote_testing": remote_testing,
                        "adaptive": adaptive,
                        "test_types": test_types,
                    }
                )
            break  # only need the Individual Test Solutions table

    return items


def scrape_detail_page(item: dict) -> dict:
    """Fetch description and job_levels from individual assessment page."""
    try:
        resp = requests.get(
            item["url"], headers=HEADERS, allow_redirects=True, timeout=15
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        div = soup.find("div", class_="product-catalogue")
        if not div:
            return item

        text = div.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        description = ""
        job_levels = []

        # Extract description (block after "Description" label)
        if "Description" in lines:
            idx = lines.index("Description")
            desc_lines = []
            for line in lines[idx + 1 :]:
                if line in ("Job levels", "Test Type:", "Remote Testing:", "Downloads"):
                    break
                desc_lines.append(line)
            description = " ".join(desc_lines)

        # Extract job levels (block after "Job levels" label)
        if "Job levels" in lines:
            idx = lines.index("Job levels")
            if idx + 1 < len(lines):
                raw = lines[idx + 1]
                job_levels = [j.strip() for j in raw.split(",") if j.strip()]

        return {**item, "description": description, "job_levels": job_levels}
    except Exception as e:
        print(f"  WARN: detail fetch failed for {item['url']}: {e}")
        return {**item, "description": "", "job_levels": []}


def scrape_all() -> list[dict]:
    print("Scraping listing pages...")
    all_items = []
    start = 0
    while True:
        print(f"  Page start={start}")
        page_items = scrape_listing_page(start)
        if not page_items:
            break
        all_items.extend(page_items)
        start += 12
        time.sleep(0.3)

    # Deduplicate by URL
    seen = set()
    unique = []
    for item in all_items:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
    print(f"Found {len(unique)} unique Individual Test Solutions")

    print("Fetching detail pages...")
    enriched = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scrape_detail_page, item): item for item in unique}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            enriched.append(result)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(unique)} done")

    return enriched


if __name__ == "__main__":
    catalog = scrape_all()
    with open("catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Saved {len(catalog)} items to catalog.json")
