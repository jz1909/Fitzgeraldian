"""
Build a Fitzgerald training corpus for nanoGPT.
"""

import argparse, csv, gzip, io, json, random, re, time, unicodedata
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = Path("fitzgerald_data")
RAW = OUT / "raw"
CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

USER, ASSIST, END = "<|user|>\n", "\n<|assistant|>\n", "\n<|end|>\n\n"

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))


# ---------------------------------------------------------------- download

def find_books():
    resp = session.get(CATALOG, timeout=60)
    resp.raise_for_status()
    rows = csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(resp.content)), encoding="utf-8"))
    books = []
    for r in rows:
        if r["Type"] == "Text" and r["Language"] == "en" and \
           "Fitzgerald, F. Scott" in r["Authors"]:
            bid = int(r["Text#"])
            books.append({"id": bid, "title": r["Title"],
                          "formats": {"text/plain; charset=utf-8": TEXT_URL.format(id=bid)}})
    return books


def strip_gutenberg(text):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*(.*?)\*\*\* ?END OF", text, re.S)
    text = m.group(1) if m else text
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- normalize

CHAR_MAP = {
    "“": '"', "”": '"', "‘": "'", "’": "'",   # curly -> straight quotes
    "…": "...", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe",
    " ": "", " ": " ", " ": " ",   # hair / thin / no-break spaces
    "_": "",                                      # Gutenberg italics markers
}


def normalize(text):
    """Shrink the character vocabulary and remove Gutenberg formatting noise."""
    text = re.sub(r"\[Transcriber.*?\]", "", text, flags=re.S | re.I)
    text = re.sub(r"^Produced by.*?\n\n", "", text, flags=re.S)
    for a, b in CHAR_MAP.items():
        text = text.replace(a, b)
    # strip accents: é -> e, ç -> c, ...
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

    paras = []
    for p in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        if not lines:
            continue
        if all(set(l) <= {"*", " "} for l in lines):   # scene break like "*   *   *"
            paras.append("* * *")
            continue
        para = " ".join(lines)                         # undo Gutenberg's ~70-col hard wrap
        paras.append(re.sub(r" {2,}", " ", para))
    return "\n\n".join(paras)


def download(books):
    RAW.mkdir(parents=True, exist_ok=True)
    seen, texts = set(), []
    for b in sorted(books, key=lambda b: b["id"]):
        title = b["title"].split(":")[0].strip()
        if title.lower() in seen:            # skip duplicate editions
            print(f"  skip duplicate: {b['title']}")
            continue
        url = next((u for k, u in b["formats"].items() if k.startswith("text/plain")), None)
        if not url:
            continue
        seen.add(title.lower())
        resp = session.get(url, timeout=60)
        resp.encoding = "utf-8"
        text = strip_gutenberg(resp.text)
        safe = re.sub(r"[^\w]+", "_", title)[:50]
        (RAW / f"{b['id']}_{safe}.txt").write_text(text, encoding="utf-8")
        print(f"  {b['id']:>6}  {len(text)/1e3:7.0f} KB  {b['title']}")
        texts.append(text)
        time.sleep(1)                         
    return texts

def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    print("Finding books...")
    books = find_books()
    print(f"Found {len(books)} entries. Downloading:")
    texts = download(books)

    corpus = "\n\n".join(normalize(t) for t in texts)
    (OUT / "input.txt").write_text(corpus, encoding="utf-8")
    print(f"\ninput.txt: {len(corpus)/1e6:.2f} MB, {len(set(corpus))} unique chars")

if __name__ == "__main__":
    main()