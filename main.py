"""
Step 2 of Backbench's RAG core: ask questions and watch what comes back.

Usage:
    python main.py test_material/your_file.pdf

This wires ingest -> chunk -> store together, then drops into a question loop
that prints the raw retrieved chunks, page numbers and all.

There is no Claude call and no teaching here, on purpose. The only claim being
tested is that a question pulls back the right section of the material — and if
retrieval is wrong, no amount of clever prompting downstream will save it.
Better to see the raw chunks now and trust them later. Grounded answers and
mistake-catching come next, in teach.py.
"""

import sys

from chunk import chunk_pages, slugify_source
from ingest import extract_text_from_pdf
from store import search_chunks, store_chunks

N_RESULTS = 3
EXIT_WORDS = {"quit", "exit", "q"}
PROMPT = "\nAsk a question (or 'quit'): "
RULE = "=" * 70


def print_results(question: str, results: list[dict]) -> None:
    """
    Print retrieved chunks in full, one block per result.

    Chunk text is shown whole rather than previewed. This is the one place
    Backbench deliberately breaks from ingest.py's 200-character previews: the
    question there was "did any text come out?", but here it's "does this chunk
    contain the answer?" — and 200 characters is about 35 words of a 175-word
    chunk. A chunk whose relevant sentence sits in the middle would look like a
    miss while being a perfect hit, and you'd end up retuning the chunker to fix
    a bug that only ever existed in the printing.
    """
    if not results:
        print("\nNothing came back. Is anything actually stored?")
        return

    print(f"\nTop {len(results)} chunk(s) for: {question}")
    print("(distance = how far a chunk sits from your question — lower is closer)")

    for rank, result in enumerate(results, start=1):
        header = f"[{rank}] page {result.get('page', '?')}   {result.get('chunk_id', 'unknown-id')}"
        distance = result.get("distance")
        if distance is not None:
            header += f"   distance {distance:.3f}"

        print(f"\n{RULE}")
        print(header)
        print(RULE)
        print(result.get("text", "").strip())

    print(f"\n{RULE}")


def run_question_loop() -> None:
    """
    Prompt for questions until the user stops, printing what retrieval returns.

    Quits on "quit", "exit", "q", Ctrl-C or Ctrl-D. Blank input just prompts
    again — a stray Enter shouldn't throw away the file we just read, chunked and
    embedded. The whole loop sits inside one try, so Ctrl-C during a slow first
    embedding call is graceful too.
    """
    try:
        while True:
            question = input(PROMPT).strip()
            if not question:
                continue
            if question.lower() in EXIT_WORDS:
                break
            print_results(question, search_chunks(question, n_results=N_RESULTS))
    except (KeyboardInterrupt, EOFError):
        print()  # step off the half-typed prompt line

    print("Bye.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        pages = extract_text_from_pdf(pdf_path)
    except FileNotFoundError as error:
        print(error)
        sys.exit(1)

    if not pages:
        print("No extractable text found. This file might be scanned/image-only")
        print("— it'll need OCR, which isn't built yet.")
        return

    chunks = chunk_pages(pages, source=slugify_source(pdf_path))

    print(f"Read {len(pages)} page(s), split into {len(chunks)} chunk(s).")
    print("Storing them now — the very first run downloads ~83 MB of embedding")
    print("model, which can look like a hang for a minute.")

    store_chunks(chunks)

    print("\nReady. Ask anything about the material. 'quit' or Ctrl-C to stop.")

    run_question_loop()


if __name__ == "__main__":
    main()
