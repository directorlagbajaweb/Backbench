"""
Step 4 of Backbench's RAG core: the terminal chat loop.

Usage:
    python main.py test_material/your_file.pdf

Setup runs once — ingest, chunk, store — and then you stay in a question loop,
typing one question after another. Each is retrieved from the stored material and
handed to teach.py, which teaches from those excerpts and only those.

Worth being precise about what this loop is. Setup happens once per run, and the
vector store and the API client both stay warm across questions, so asking ten
questions costs one setup instead of ten. But each answer is generated
independently: generate_answer() takes a question and its own excerpts and no
history at all. Self-contained questions work well; a follow-up that leans on the
previous answer ("why?", "explain that further") gets retrieved and answered as
though it were asked cold, because that's literally what happens. Carrying
conversation history is the next step, and it belongs in teach.py.

Retrieved chunks are no longer dumped in full — see format_sources() for why. To
inspect raw text, `python chunk.py <pdf>` prints every chunk in a file and
`python teach.py "<question>"` prints previews of the ones a question pulls back.
"""

import sys

from chunk import chunk_pages, slugify_source
from ingest import extract_text_from_pdf
from listen import listen
from speak import speak
from store import search_chunks, store_chunks
from teach import generate_answer

N_RESULTS = 3
MAX_HISTORY = 4  # exchanges kept, so a long session can't grow the prompt forever
FOLLOWUP_MAX_WORDS = 6  # at or below this, a question after an answer reads as a follow-up
EXIT_WORDS = {"quit", "exit", "q"}
VOICE_WORDS = {"v", "voice"}
PROMPT = "\nAsk a question — type it, or 'v' to speak it ('quit' to stop): "
RULE = "=" * 70


def format_sources(results: list[dict]) -> str:
    """
    Summarise where an answer's material came from, in one line.

    Returns a string like:
        sources: page 1 (0.292), page 5 (0.458), page 5 (0.459)

    Step 2 printed the full text of every retrieved chunk here, because proving
    retrieval was then the whole point. Now that answers are grounded and cite
    their own pages, three chunks of raw prose ahead of every reply would bury the
    answer and make a back-and-forth unreadable. Dropping provenance entirely
    would be worse though — without it you can't tell a wrong answer caused by bad
    retrieval from one caused by bad teaching, which is the first thing you need
    to know. So the pages and distances stay, on one line.
    """
    parts = []

    for result in results:
        part = f"page {result.get('page', '?')}"
        distance = result.get("distance")
        if distance is not None:
            part += f" ({distance:.3f})"
        parts.append(part)

    return "sources: " + ", ".join(parts)


def run_question_loop() -> None:
    """
    Prompt for questions until the user stops, teaching an answer to each.

    Quits on "quit", "exit", "q", Ctrl-C or Ctrl-D. Blank input just prompts
    again — a stray Enter shouldn't throw away the file we just read, chunked and
    embedded, and now also shouldn't cost an API call.

    Errors are split by whether waiting helps. A missing API key fails every
    question identically, so there's no point sitting in the loop; a rate limit or
    a server blip is worth staying open for, since the setup cost is already paid.

    Conversation memory lives here, in a plain list, for as long as the process
    runs. It's deliberately not inside teach.py: generate_answer() stays a pure
    function of what it's given, and a caller that wants no memory — the board
    server — just doesn't pass any.
    """
    history: list[dict] = []

    try:
        while True:
            question = input(PROMPT).strip()
            if not question:
                continue
            if question.lower() in EXIT_WORDS:
                break

            # Purely additive: the typed path above is untouched, and this only
            # runs when you ask for it. listen() returns None for every kind of
            # failure, and the continue sends you straight back to the prompt to
            # type instead — voice never blocks a question from being asked.
            if question.lower() in VOICE_WORDS:
                spoken = listen()
                if spoken is None:
                    continue
                print(f'heard: "{spoken}"')
                question = spoken

            # Giving history to generate_answer() lets it understand "why?", but
            # retrieval still runs on the literal word. Measured: "why?" alone
            # retrieves at distance 0.976, where a completely unrelated question
            # scores 0.988 — so the model understood the follow-up and then had
            # nothing relevant to answer it from. Prepending the previous question
            # to the *search query* puts it back on the right section: 3 of 3
            # correct chunks instead of 0 of 3.
            #
            # Only the search query is widened. generate_answer() still gets the
            # bare question, so the model is never told the student asked
            # something they didn't.
            search_query = question
            if history and len(question.split()) <= FOLLOWUP_MAX_WORDS:
                search_query = f"{history[-1]['question']} {question}"
                print(f"\n(reading as a follow-up to: {history[-1]['question']})")

            results = search_chunks(search_query, n_results=N_RESULTS)
            if not results:
                print("\nNothing came back. Is anything actually stored?")
                continue

            print(f"\n{format_sources(results)}")
            print(RULE)

            try:
                answer = generate_answer(question, results, history=history)
            except RuntimeError as error:
                # Configuration, not weather — a missing key won't fix itself.
                print(error)
                print(RULE)
                break
            except Exception as error:
                # Broad on purpose, and provider-agnostic on purpose: main.py
                # never imports the API SDK, so swapping teach.py between Gemini
                # and Claude needs no change here. Ctrl-C still escapes, because
                # KeyboardInterrupt is a BaseException and isn't caught by this.
                #
                # Read .message and .code by duck-typing rather than importing the
                # provider's exception classes. Both SDKs expose them, and plain
                # str() on a Gemini APIError stringifies the entire JSON payload —
                # several hundred characters of quota metadata in the middle of a
                # conversation.
                detail = getattr(error, "message", None) or str(error)
                code = getattr(error, "code", None)
                print(f"That question didn't get an answer"
                      f"{f' ({code})' if code else ''}: {detail}")
                print(RULE)
                continue

            print(answer)
            print(RULE)

            # Remember this exchange so the next question can be a follow-up.
            # Only successful answers go in — an error message is not something
            # the next turn should try to interpret "why?" against. Trimmed to the
            # last few exchanges so a long session doesn't grow the prompt without
            # limit. Typed and spoken questions land here identically, because by
            # this point both are just `question`.
            history.append({"question": question, "answer": answer})
            del history[:-MAX_HISTORY]

            # Printing comes first and speaking second, so the answer is on
            # screen whatever happens to the audio. speak() swallows its own
            # failures for the same reason — voice is additive here, never a
            # dependency of the loop.
            speak(answer)
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
