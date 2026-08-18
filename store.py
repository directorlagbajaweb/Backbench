"""
Step 2 of Backbench's RAG core: store chunks so they can be found by meaning.

Usage:
    python store.py test_material/your_file.pdf

Keeps a Chroma collection on disk in ./chroma_db. Chroma does the embedding
itself with a small model it downloads on first use, so nothing here needs an
API key — the Anthropic key is only for the teaching step later. Heads up that
the first run stalls for a minute while ~83 MB of model comes down; it looks
like a hang and isn't one, and it has no offline fallback.

Re-running the pipeline on the same PDF is idempotent: chunk IDs are stable and
already carry the source filename, so upsert replaces each row in place. Two
different PDFs accumulate side by side rather than overwriting each other.

Deferred: forgetting a document, and re-embedding after the chunker changes.
Upsert replaces chunks but never removes them, so if you retune chunk.py and a
PDF yields *fewer* chunks than before, the old high-numbered ones stay in the
collection and can still be retrieved. The fix when that day comes is one line —
collection.delete(where={"source": source}) before upserting — which is why
source is kept in metadata now.
"""

import sys
from pathlib import Path

import chromadb

from chunk import chunk_pages, slugify_source
from ingest import extract_text_from_pdf

DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "backbench"
BATCH_SIZE = 200  # documents embedded per call — a textbook shouldn't go in one gulp

# Chroma caches one client per path and refuses a second one built with
# different settings, so both of these are made once and reused.
_client = None
_collection = None


def get_client():
    """
    Open (once) the Chroma client that owns ./chroma_db.

    Deliberately does not pass Settings. In current Chroma the telemetry client
    is a no-op, so Settings(anonymized_telemetry=False) silences nothing — and
    passing Settings here but not elsewhere is exactly what triggers "An
    instance of Chroma already exists ... with different settings".
    """
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)

    return _client


def get_collection():
    """
    Open (once) the collection everything is stored in, creating it if needed.

    The distance metric is set explicitly even though cosine is already what the
    default embedding model asks for: it costs nothing and it records which
    metric the stored vectors assume. Note this only applies at creation — on an
    existing collection Chroma ignores it silently, so changing the metric later
    means deleting ./chroma_db, not editing this line.
    """
    global _collection

    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    return _collection


def store_chunks(chunks: list[dict]) -> None:
    """
    Write chunks into the collection, replacing any earlier copy of them.

    Each chunk looks like:
        {"chunk_id": "week-3-notes-p4-c2", "page": 4,
         "text": "...", "source": "week-3-notes"}

    Uses upsert rather than add so re-running the pipeline on the same PDF is a
    no-op instead of doubling everything. Letting duplicates in would be worse
    than it sounds: with n_results=3, three copies of one chunk can fill all
    three slots, so the teaching step gets one chunk of context while believing
    it has three.
    """
    if not chunks:
        return

    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append(
            {
                "source": chunk.get("source", "unknown"),
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            }
        )

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def search_chunks(query: str, n_results: int = 3) -> list[dict]:
    """
    Find the chunks closest in meaning to the query.

    Returns a list of dicts like:
        {"chunk_id": "week-3-notes-p4-c2", "page": 4, "source": "week-3-notes",
         "text": "...", "distance": 0.21}

    Nearest first. "distance" is a cosine distance, so *smaller is closer* — it
    is not a similarity score and must never be shown to anyone as one.

    Note that Chroma always returns the nearest n_results, however far away they
    are; there is no notion of "no match". Judging whether the material actually
    covers a question means looking at the distance, which is why it's returned.
    """
    collection = get_collection()

    stored = collection.count()
    if stored == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, stored),
    )

    # Chroma nests results one level per query. We only ever send one query, so
    # everything is read out of index 0.
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    hits = []
    for record_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        hits.append(
            {
                "chunk_id": metadata.get("chunk_id", record_id),
                "page": metadata.get("page"),
                "source": metadata.get("source"),
                "text": text,
                "distance": distance,
            }
        )

    return hits


def main():
    if len(sys.argv) != 2:
        print("Usage: python store.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        print("No extractable text found. This file might be scanned/image-only")
        print("— it'll need OCR, which isn't built yet.")
        return

    chunks = chunk_pages(pages, source=slugify_source(pdf_path))

    print(f"Storing {len(chunks)} chunk(s) from {len(pages)} page(s)...")
    print("On the very first run this downloads ~83 MB of embedding model,")
    print("which can look like a hang for a minute.\n")

    store_chunks(chunks)
    print(f"Collection now holds {get_collection().count()} chunk(s).\n")

    query = "what is this document about?"
    print(f'Smoke-test search for: "{query}"')
    print("(distance = how far a chunk sits from the query — lower is closer)\n")

    for hit in search_chunks(query):
        preview = hit["text"][:200]
        print(f"--- {hit['chunk_id']} (page {hit['page']}, "
              f"distance {hit['distance']:.3f}) ---")
        print(preview + ("..." if len(hit["text"]) > 200 else ""))
        print()


if __name__ == "__main__":
    main()
