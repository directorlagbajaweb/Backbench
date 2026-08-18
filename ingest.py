"""
Step 1 of Backbench's RAG core: get real text out of a real file.

Usage:
    python ingest.py test_material/your_file.pdf

This only handles native-text PDFs for now (PDFs where the text can be
selected/copied normally). Scanned pages or photos of notes need OCR,
which comes later once this path is working end to end.
"""

import sys
from pathlib import Path

import pymupdf


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, page by page.

    Returns a list of dicts like:
        {"page": 1, "text": "..."}

    Keeping page numbers attached now matters later — when the bot answers
    a question, we want to be able to say *which page* it came from.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Can't find file: {pdf_path}")

    doc = pymupdf.open(path)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({"page": page_number, "text": text})

    doc.close()
    return pages


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        print("No extractable text found. This file might be scanned/image-only")
        print("— it'll need OCR, which isn't built yet.")
        return

    print(f"Extracted text from {len(pages)} page(s):\n")
    for page in pages:
        preview = page["text"][:200].replace("\n", " ")
        print(f"--- Page {page['page']} ---")
        print(preview + ("..." if len(page["text"]) > 200 else ""))
        print()


if __name__ == "__main__":
    main()
