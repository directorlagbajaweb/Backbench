"""
Step 2 of Backbench's RAG core: cut each page into searchable pieces.

Usage:
    python chunk.py test_material/your_file.pdf

Chunks are built one page at a time, never across a page boundary, so every
chunk can name the page it came from — that citation is the whole promise of a
bot that only speaks from your material.

Chunk size is capped at 175 words for a reason that isn't obvious: Chroma's
default embedding model reads at most 256 tokens (~180 words) and *silently
truncates* anything longer. A 400-word chunk would store fine and print fine
while only its opening was ever searchable, so the sizes here are chosen to fit
that window whole rather than to look generous.

Splitting is word-based and deliberately dumb: no sentence detection, no
heading awareness, and no repair of line-wrapped hyphens ("thermo-\\ndynamics"
stays as "thermo- dynamics", because the naive fix also turns "well-\\nknown"
into "wellknown" and telling those apart needs a dictionary we don't have).
Smarter boundaries come later, once retrieval is proven end to end.
"""

import re
import sys
from pathlib import Path

from ingest import extract_text_from_pdf

CHUNK_SIZE = 175      # words per chunk — fits inside the embedder's 256-token window
OVERLAP = 30          # words repeated from the end of the previous chunk
MIN_TAIL_WORDS = 45   # a trailing piece smaller than this is folded backwards
MAX_WORDS = 190       # hard ceiling — a fold must never cross the token window


def slugify_source(name: str) -> str:
    """
    Turn a file path into a short, ID-safe tag.

    Returns a string like:
        "week-3-notes"

    Chunk IDs carry this as a prefix so two different PDFs can each have a
    page 1 chunk 1 without fighting over the same Chroma document ID.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")
    return slug or "doc"


def split_into_words(text: str) -> list[str]:
    """
    Split page text into words on any run of whitespace.

    Returns a list like:
        ["Thermodynamics", "is", "the", "study", "of", "heat", ...]

    This is the only place tokenising happens, so every word count downstream is
    measured in the same unit — and if we ever need to preserve line breaks,
    only this function and build_chunk_text have to change.
    """
    return text.split()


def window_bounds(
    total_words: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    min_tail: int = MIN_TAIL_WORDS,
    max_words: int = MAX_WORDS,
) -> list[tuple[int, int]]:
    """
    Work out the (start, end) word slices for one page, overlap included.

    Returns a list like:
        [(0, 175), (145, 314)]

    Pure integer arithmetic, no text, so the awkward cases — a page shorter than
    one chunk, a page ending in a 20-word runt, an overlap wider than the chunk
    itself — can be tested without a PDF anywhere near them.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")
    if total_words <= 0:
        return []
    if total_words <= chunk_size:
        return [(0, total_words)]

    stride = chunk_size - overlap
    bounds = []
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        bounds.append((start, end))
        if end == total_words:
            break  # stop here, or the next window is a subset of this one
        start += stride

    # Fold a stubby last chunk into the one before it — a 20-word chunk is
    # retrieval noise that can win on a keyword and then teach nothing. But
    # never fold past max_words, or the merged chunk gets silently truncated by
    # the embedder, which is the exact problem these sizes exist to avoid.
    if len(bounds) > 1:
        previous_start, previous_end = bounds[-2]
        _, last_end = bounds[-1]
        new_words = last_end - previous_end
        if new_words < min_tail and last_end - previous_start <= max_words:
            bounds[-2] = (previous_start, last_end)
            bounds.pop()

    return bounds


def build_chunk_text(words: list[str], start: int, end: int) -> str:
    """
    Join a word slice back into readable text.

    Returns a string like:
        "Thermodynamics is the study of heat ..."

    Single spaces, so the text the bot quotes back never carries the ragged line
    breaks PyMuPDF picks up from the page layout.
    """
    return " ".join(words[start:end])


def make_chunk_id(source: str, page: int, index: int) -> str:
    """
    Build a stable, readable ID for one chunk.

    Returns a string like:
        "week-3-notes-p4-c2"

    Chroma needs these globally unique, and a human debugging a bad retrieval
    needs to read the page straight off the ID. Numbering per page rather than
    per document also means editing page 4 renumbers only page 4, instead of
    shifting every ID in the document and orphaning the old ones.
    """
    return f"{source}-p{page}-c{index}"


def chunk_page(page: dict, source: str = "doc") -> list[dict]:
    """
    Split a single page dict into overlapping chunks.

    Returns a list of dicts like:
        {"chunk_id": "week-3-notes-p4-c2", "page": 4,
         "text": "...", "source": "week-3-notes"}

    A short page (a slide, a title page) comes back as one small chunk on
    purpose — that page is already a single idea, and dropping it would mean the
    bot can never quote it.
    """
    words = split_into_words(page["text"])
    chunks = []

    for index, (start, end) in enumerate(window_bounds(len(words)), start=1):
        chunks.append(
            {
                "chunk_id": make_chunk_id(source, page["page"], index),
                "page": page["page"],
                "text": build_chunk_text(words, start, end),
                "source": source,
            }
        )

    return chunks


def chunk_pages(pages: list[dict], source: str = "doc") -> list[dict]:
    """
    Split every page of a document into chunks, in reading order.

    Returns a list of dicts like:
        {"chunk_id": "week-3-notes-p4-c2", "page": 4,
         "text": "...", "source": "week-3-notes"}

    This is the exact shape store.py hands to Chroma: text to embed, chunk_id as
    the document ID, page as the metadata that lets an answer cite itself, and
    source so one document can later be filtered or forgotten on its own.
    """
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page, source))
    return chunks


def main():
    if len(sys.argv) != 2:
        print("Usage: python chunk.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        print("No extractable text found. This file might be scanned/image-only")
        print("— it'll need OCR, which isn't built yet.")
        return

    chunks = chunk_pages(pages, source=slugify_source(pdf_path))
    counts = [len(split_into_words(chunk["text"])) for chunk in chunks]

    print(f"Split {len(pages)} page(s) into {len(chunks)} chunk(s).")
    print(f"Words per chunk: smallest {min(counts)}, largest {max(counts)}, "
          f"average {sum(counts) // len(counts)}")

    oversized = [count for count in counts if count > MAX_WORDS]
    if oversized:
        print(f"WARNING: {len(oversized)} chunk(s) exceed {MAX_WORDS} words and")
        print("will be silently truncated when embedded. That's a bug in the")
        print("chunking arithmetic, not something to shrug at.")
    print()

    for chunk, count in zip(chunks, counts):
        preview = chunk["text"][:200]
        print(f"--- {chunk['chunk_id']} (page {chunk['page']}, {count} words) ---")
        print(preview + ("..." if len(chunk["text"]) > 200 else ""))
        print()


if __name__ == "__main__":
    main()
