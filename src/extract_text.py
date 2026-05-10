"""
Extract markdown text from all PDFs in data/selected_contracts/ for chunking and embedding.

Run from project root:
    python3 data/extract_text.py
"""

import csv
from pathlib import Path

import pymupdf4llm
import fitz

CONTRACTS_DIR = Path("data/selected_contracts")
TEXT_DIR      = Path("data/selected_contracts_text")
METADATA_CSV  = Path("extraction_metadata.csv")


def extract_pdf(pdf_path: Path) -> dict:
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        with fitz.open(str(pdf_path)) as doc:
            page_count = len(doc)
        return {"text": md_text, "page_count": page_count, "char_count": len(md_text), "error": ""}
    except Exception as exc:
        return {"text": "", "page_count": 0, "char_count": 0, "error": str(exc)}


def main():
    TEXT_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(CONTRACTS_DIR.glob("*.pdf"))
    total = len(pdf_files)
    print(f"Found {total} PDFs in {CONTRACTS_DIR}/\n")

    metadata_rows = []
    failed = []

    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx:>3}/{total}] {pdf_path.name} ... ", end="", flush=True)

        result = extract_pdf(pdf_path)

        out_path = TEXT_DIR / (pdf_path.stem + ".txt")
        out_path.write_text(result["text"], encoding="utf-8")

        if result["error"]:
            print(f"FAILED | {result['error'][:80]}")
            failed.append(pdf_path.name)
        else:
            print(f"{result['char_count']:,} chars | {result['page_count']} pages")

        metadata_rows.append({
            "filename":   pdf_path.name,
            "page_count": result["page_count"],
            "char_count": result["char_count"],
            "error":      result["error"],
        })

    with METADATA_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "page_count", "char_count", "error"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDone. {total - len(failed)}/{total} extracted successfully.")
    print(f"Text files → {TEXT_DIR}/")
    print(f"Metadata  → {METADATA_CSV}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
