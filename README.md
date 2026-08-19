# Backbench

A teaching bot that learns from *your* course material — PDFs, notes, textbook
pages — and teaches strictly from what you give it, not generic web knowledge.

## Phase 1 goal (where we are now)

Prove the core works before building any UI, voice, or visuals:

1. Ingest a real PDF or set of notes
2. Chunk and store it so it can be searched by meaning
3. Retrieve the right chunk for a question
4. Teach from that chunk only, and catch mistakes in examples

Everything in Phase 1 runs from the terminal. No frontend yet.

## Setup

```bash
python3.13 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Use Python 3.13 (or 3.12). Chroma's dependency tree is only tested up to 3.13,
and there's an open report of a native crash on macOS ARM64 under 3.14 — not
worth debugging a segfault on a learning project. On macOS:
`brew install python@3.13`.

Add your API key (Step 3 onwards needs it):

```bash
cp .env.example .env
# then edit .env and paste your key in
```

## Step 1 — ingest a file

Drop a real PDF into `test_material/`, then run:

```bash
python ingest.py test_material/your_file.pdf
```

This should print out the extracted text so you can sanity-check it before
moving on to chunking and storage.

## Step 2 — chunk, store, and prove retrieval works

Three scripts, each runnable on its own so you can see where a problem is:

```bash
python chunk.py test_material/your_file.pdf   # just the splitting
python store.py test_material/your_file.pdf   # splitting + embedding + one test search
python main.py  test_material/your_file.pdf   # the whole thing, then ask questions
```

`main.py` runs ingest → chunk → store, then drops into a loop where you type a
question and it prints the raw chunks that came back, with page numbers. No
Claude call yet — the point of this step is to confirm with your own eyes that a
question retrieves the *right section* before any answering logic exists.

Two things worth knowing on first run:

- **It will look like it hung.** The first search downloads ~83 MB of embedding
  model to `~/.cache/chroma/`. That's a one-off, and it needs an internet
  connection — there's no offline fallback.
- **Chunks are capped at ~175 words**, which is smaller than typical RAG advice.
  Chroma's default embedding model reads only the first 256 tokens of a document
  and silently truncates the rest, so bigger chunks would store and print fine
  while being half-invisible to search.

Re-running on the same PDF is safe and won't duplicate anything. Different PDFs
accumulate in the same collection, in `./chroma_db` (gitignored).

Watch the `distance` numbers: lower means closer. Try a question your material
*doesn't* cover — you'll still get three chunks back, because Chroma always
returns the nearest ones, just with visibly worse distances. That gap is what
Step 3 will use to say "your notes don't cover that" instead of making something
up.

## Step 3 — teach from the retrieved chunks

This is the first step that needs an API key. `teach.py` currently calls
Google's Gemini API, whose free tier works without billing set up:

```bash
cp .env.example .env    # then paste your Gemini key in as GEMINI_API_KEY
python teach.py "what research design did the study use?"
```

Free keys come from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Switching back to Claude later means editing `MODEL`, `get_client()`, and the
request handling in `generate_answer()` — the system prompt and everything else
in the file is provider-agnostic, and `anthropic` stays pinned in
`requirements.txt` for that.

`teach.py` prints the chunks it retrieved and then the answer Claude taught from
them, so when an answer looks wrong you can see immediately whether retrieval or
teaching was at fault.

The system prompt does three things worth knowing about:

- **It only teaches from the retrieved chunks**, and says "your material doesn't
  cover this" rather than filling gaps from general knowledge — even when it
  knows the answer. Otherwise you can't tell which half of an answer came from
  your course.
- **It judges relevance itself.** Retrieval always returns its closest matches,
  however far off they are, and Step 2 showed distance scores don't separate a
  wrong answer from no answer reliably. So distances are passed in as a hint for
  Claude to weigh, not used as a filter before it ever sees the chunks.
- **It catches mistakes.** If a worked example in your material contains a clear
  factual, mathematical, or logical error, it says so and teaches the corrected
  version instead of repeating the error as fact.

Not wired into `main.py` yet — that's Step 4.

## Roadmap

- [x] Step 1 — `ingest.py`: extract text from a PDF
- [x] Step 2 — `chunk.py` + `store.py` + `main.py`: split, store, prove retrieval
- [x] Step 3 — `teach.py`: grounded teaching via Claude
- [ ] Step 4 — `main.py`: full chat loop wired to teach.py
