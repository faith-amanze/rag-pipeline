"""fetch_kb.py - download a starter knowledge base (tech, misinformation, privacy) into kb/

Usage: python fetch_kb.py
Then:  python ingest.py --kb kb
"""
import sys
from pathlib import Path
import requests

SOURCES = {
    # Reuters Institute Digital News Report 2026 (CC BY 4.0): trust, misinformation, platforms, AI chatbots
    "reuters_dnr_2026_full_report.pdf":
        "https://reutersinstitute.politics.ox.ac.uk/sites/default/files/2026-06/DNR%202026%20FINAL_2.pdf",
    "reuters_dnr_2026_executive_summary.html":
        "https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2026/dnr-executive-summary",
    # Pew Research Center: privacy and personal data
    "pew_2023_key_findings_data_privacy.html":
        "https://www.pewresearch.org/short-reads/2023/10/18/key-findings-about-americans-and-data-privacy/",
    "pew_2019_key_takeaways_privacy_surveillance.html":
        "https://www.pewresearch.org/short-reads/2019/11/15/key-takeaways-on-americans-views-about-privacy-surveillance-and-data-sharing/",
    "pew_2020_opting_out_privacy_concerns.html":
        "https://www.pewresearch.org/short-reads/2020/04/14/half-of-americans-have-decided-not-to-use-a-product-or-service-because-of-privacy-concerns",
    "pew_2019_how_americans_think_about_privacy.html":
        "https://www.pewresearch.org/internet/2019/11/15/how-americans-think-about-privacy-and-the-vulnerability-of-their-personal-data/",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; RAG demo)"}


def main():
    kb = Path("kb")
    kb.mkdir(exist_ok=True)
    ok = 0
    for name, url in SOURCES.items():
        dest = kb / name
        if dest.exists():
            print(f"skip   {name} (already there)")
            ok += 1
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"saved  {name} ({len(r.content) / 1024:.0f} KB)")
            ok += 1
        except Exception as e:
            print(f"FAILED {name}: {e}\n       download it manually from {url}", file=sys.stderr)
    print(f"\n{ok}/{len(SOURCES)} files in kb/")


if __name__ == "__main__":
    main()
