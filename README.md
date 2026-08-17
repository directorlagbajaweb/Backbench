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
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add your Anthropic API key:

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

## Roadmap

- [x] Step 1 — `ingest.py`: extract text from a PDF
- [ ] Step 2 — `chunk.py` + `store.py`: split and store in Chroma
- [ ] Step 3 — `teach.py`: retrieval + grounded teaching via Claude
- [ ] Step 4 — `main.py`: terminal chat loop
