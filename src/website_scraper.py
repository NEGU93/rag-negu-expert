import json
import time
from datetime import datetime
from pathlib import Path
from langchain.schema import Document
from playwright.sync_api import sync_playwright
from src.logger_init import logger

BASE_URL = "https://negu93.github.io"
ROUTES = [
    # "home",
    "timeline",
    "phd",
    # "languages",
    # "chat_llm",
]
OUTPUT_DIR = Path("data/website")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def auto_scroll(page, pause_time: float = 0.3, max_scrolls: int = 50):
    """
    Scrolls slowly down the page until no more new content is loaded.
    Returns when scrolled to bottom or max_scrolls reached.
    """
    logger.info("🔽  Scrolling page to load dynamic content...")
    last_height = page.evaluate("document.body.scrollHeight")
    for i in range(max_scrolls):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(pause_time)
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            logger.info(f"✅  Finished scrolling after {i} iterations.")
            break
        last_height = new_height


def extract_page_text(page):
    """Extract all visible text on the page."""
    return page.inner_text("body").strip()


def parse_date(date_str: str):
    try:
        return datetime.fromisoformat(date_str).isoformat()
    except:
        return date_str


def scrape_timeline(page, url: str):
    """
    Extract structured timeline events (IEvent) from the Angular timeline page.
    Returns a list of langchain.schema.Document objects.
    """

    def safe_inner_text(parent, selector):
        """Return inner_text() of selector or empty string if not found."""
        node = parent.query_selector(selector)
        return node.inner_text().strip() if node else ""

    def safe_attribute(parent, selector, attr):
        """Return attribute value of selector or empty string if not found."""
        node = parent.query_selector(selector)
        return node.get_attribute(attr) if node else ""

    logger.info("🔽 Scrolling page to load all timeline events...")
    auto_scroll(page, pause_time=0.3, max_scrolls=50)

    # Select all timeline blocks
    event_elements = page.query_selector_all("#cd-timeline .cd-timeline-block")
    logger.info(f"Found {len(event_elements)} timeline events.")
    documents = []

    for i, el in enumerate(event_elements, start=1):
        try:
            role = safe_inner_text(el, "h2")
            department = safe_inner_text(el, "h3:nth-of-type(1)")
            enterprise = safe_inner_text(el, "h3:nth-of-type(2)")
            description = safe_inner_text(el, ".editor p")

            # Dates are displayed as "MM/yyyy - MM/yyyy" or "MM/yyyy - Present"
            date_text = safe_inner_text(el, ".cd-date")
            start_date, end_date = (None, None)
            if "-" in date_text:
                parts = date_text.split("-")
                start_date = parts[0].strip()
                end_date = parts[1].strip() if len(parts) > 1 else None
            else:
                start_date = date_text.strip()

            logo_url = safe_attribute(el, ".enterprise-logo", "src")

            certificate_url = safe_attribute(el, ".editor a", "href")

            # Construct the event dict
            event = {
                "role": role,
                "department": department or None,
                "enterprise": enterprise,
                "description": description,
                "startDate": start_date,
                "endDate": end_date,
                "logoUrl": logo_url,
                "certificate": certificate_url or None,
                "tags": [],
                "eventName": "experience",  # optional, can be customized per event
            }

            documents.append(
                Document(
                    page_content=str(event),  # store the dict as a string
                    metadata={
                        "source": url,
                        "source_type": "timeline",
                        **event,
                    },
                )
            )

        except Exception as e:
            logger.error(f"⚠️ Failed to parse event #{i}: {e}")
            continue

    logger.info(f"✅ Parsed {len(documents)} timeline events successfully.")
    return documents


def scrape_website(base_url: str = BASE_URL, routes: list[str] = ROUTES):
    """Render each route (including lazy-loaded scroll content)."""
    docs: list[Document] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for route in routes:
            url = f"{base_url}/{route}"
            logger.info(f"\n🕸️  Visiting {url} ...")

            try:
                page.goto(url, wait_until="networkidle", timeout=90000)
                time.sleep(1.5)

                if "timeline" in route:
                    timeline_docs = scrape_timeline(page, url)
                    docs.extend(timeline_docs)

                    # Save timeline events to a JSON for inspection
                    timeline_json_file = OUTPUT_DIR / f"{route}.json"
                    with open(timeline_json_file, "w", encoding="utf-8") as f:
                        json.dump(
                            [eval(d.page_content) for d in timeline_docs],
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    logger.info(
                        f"✅ Saved {len(timeline_docs)} timeline events to {timeline_json_file}"
                    )

                else:
                    text = extract_page_text(page)
                    if not text:
                        logger.warning(f"⚠️ No text extracted from {url}")
                        continue
                    file_path = OUTPUT_DIR / f"{route}.txt"
                    file_path.write_text(text, encoding="utf-8")
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": url, "source_type": "website"},
                        )
                    )
                    logger.info(
                        f"✅ Saved {file_path.name} ({len(text)} chars)"
                    )

            except Exception as e:
                logger.error(f"❌  Error scraping {url}: {e}")

                browser.close()

    return docs


if __name__ == "__main__":
    logger.info("Starting full website scrape...\n")
    docs = scrape_website()
    logger.info(f"\n📁 Saved {len(docs)}")
