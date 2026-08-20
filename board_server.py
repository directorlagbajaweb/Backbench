"""
Step 7 of Backbench's RAG core: serve the live board to a browser.

Usage:
    python board_server.py                          # use what's already stored
    python board_server.py "test_material/x.pdf"    # ingest that file first

Then open http://127.0.0.1:8000 in a browser.

This is additive: main.py still runs the terminal loop exactly as before, and
nothing here changes chunk.py, store.py or teach.py. It imports the same
generate_answer() the terminal uses, so the board and the terminal cannot drift
apart in what they teach.

Why the standard library rather than FastAPI: this needs one JSON endpoint and one
static file. FastAPI's advantages — response streaming, schema validation, async
concurrency — would earn their keep if the model output were streamed token by
token, but generate_answer() returns a finished string, so there is nothing to
stream. http.server does the job with no new dependency and no wheel to worry
about on Python 3.14.

Which also decides where the writing animation lives: the browser. The server
sends the whole answer once, and board.html reveals it word by word. Real
token-by-token streaming would mean teach.py growing a streaming variant, which
belongs in a later step; animating client-side has the side benefit that the
writing speed stays smooth regardless of how the network behaves.

Bound to 127.0.0.1 on purpose. This endpoint spends API quota on whoever calls
it, so it should not be reachable from the network.
"""

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chunk import chunk_pages, slugify_source
from ingest import extract_text_from_pdf
from listen import transcribe
from speak import speak
from store import get_collection, search_chunks, store_chunks
from teach import generate_answer

HOST = "127.0.0.1"
PORT = 8000
N_RESULTS = 3
BOARD_PATH = Path(__file__).resolve().parent / "board.html"
MAX_QUESTION_CHARS = 2000
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# One thing speaks at a time. The board only asks for speech once its own writing
# animation has finished, and starting a new question abandons that animation, so
# two answers rarely contend — but if they do, the newer one waits rather than
# talking over the older one, and anything superseded while it waited is dropped.
#
# Two locks, deliberately. _speaking is held for the whole utterance, which can be
# twenty seconds; _counter only ever guards an integer. Sharing one lock would
# make a new request block behind the speech already playing, and that request is
# an HTTP handler.
_speaking_lock = threading.Lock()
_counter_lock = threading.Lock()
_speak_generation = 0

# Browser audio arrives as opus-in-webm (Chrome, Firefox) or aac-in-mp4 (Safari).
# The suffix is only a hint — PyAV, already present as a faster-whisper
# dependency, sniffs the container itself.
AUDIO_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
}


def answer_question(question: str) -> dict:
    """
    Retrieve and teach one question, as a JSON-ready dict.

    Returns either
        {"answer": "...", "sources": [{"page": 4, "distance": 0.731, ...}]}
    or
        {"error": "..."}

    Errors come back as data rather than as an exception, so the board can print
    what went wrong in the page instead of the fetch failing and leaving the
    reader looking at a blank board.
    """
    chunks = search_chunks(question, n_results=N_RESULTS)
    if not chunks:
        return {"error": "Nothing is stored yet — ingest a PDF first."}

    try:
        answer = generate_answer(question, chunks)
    except RuntimeError as error:
        return {"error": str(error)}
    except Exception as error:
        # Duck-typed for the same reason main.py does it: this file stays
        # provider-agnostic, and plain str() on a Gemini error dumps the whole
        # JSON payload.
        detail = getattr(error, "message", None) or str(error)
        code = getattr(error, "code", None)
        return {"error": f"{code + ': ' if code else ''}{detail}"}

    return {
        "answer": answer,
        "sources": [
            {
                "page": chunk.get("page"),
                "distance": chunk.get("distance"),
                "chunk_id": chunk.get("chunk_id"),
            }
            for chunk in chunks
        ],
    }


def speak_in_background(text: str) -> None:
    """
    Say text aloud without making the browser wait for it.

    speak() blocks for as long as the speech lasts — twenty seconds is normal for
    a taught answer — so calling it inside a request handler would hold the
    connection open that whole time. This hands it to a daemon thread and returns
    at once.

    The audio comes out of the machine running this server. That's the right
    behaviour here precisely because the board is bound to 127.0.0.1: the browser
    and the speakers are the same computer.
    """
    global _speak_generation

    with _counter_lock:
        _speak_generation += 1
        mine = _speak_generation

    def run() -> None:
        # Wait for any speech already playing, and only then decide whether this
        # one is still worth saying. Checking before the wait would be useless:
        # what makes an utterance stale is another arriving while it queues.
        with _speaking_lock:
            with _counter_lock:
                if mine != _speak_generation:
                    return

            # speak() does the markdown stripping and the maths-into-words work,
            # so the board sends the raw answer and this stays the only place
            # that knows how an answer should be read aloud. Called while holding
            # _speaking_lock, so two answers can never overlap.
            speak(text)

    threading.Thread(target=run, daemon=True).start()


def transcribe_upload(audio: bytes, content_type: str) -> dict:
    """
    Turn an uploaded recording into text.

    Returns {"text": "..."} or {"error": "..."}.

    Reuses listen.transcribe() exactly as the terminal uses it — same model, same
    settings. Only the recording half differs: listen.record_until_enter() waits
    on Enter from the server's own stdin, which no browser can provide, so the
    page records with MediaRecorder and posts the result here instead.
    """
    suffix = AUDIO_SUFFIXES.get(content_type.split(";")[0].strip(), ".bin")
    path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio)
            path = handle.name

        text = transcribe(path)
        if not text:
            return {"error": "No speech found in that recording."}
        return {"text": text}

    except Exception as error:
        # The detail goes to the server log, not the page: decoder errors quote
        # the temp file path, and there's no reason to show a browser a path
        # inside /var/folders.
        print(f"  transcription failed: {type(error).__name__}: {error}")
        return {"error": "That recording couldn't be read as audio. "
                         "Try again, or type the question."}
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


class BoardHandler(BaseHTTPRequestHandler):
    """Serves the board page and answers one question at a time."""

    server_version = "Backbench/1.0"

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/board.html"):
            try:
                body = BOARD_PATH.read_bytes()
            except OSError:
                self.send_error(500, f"Cannot read {BOARD_PATH.name}")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "Not found")

    def read_body(self, limit: int) -> bytes | None:
        """Read the request body, or None if the length is missing or too big."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None

        if length <= 0 or length > limit:
            return None

        return self.rfile.read(length)

    def do_POST(self) -> None:
        if self.path == "/listen":
            self.handle_listen()
            return

        if self.path == "/speak":
            self.handle_speak()
            return

        if self.path != "/ask":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Bad Content-Length."}, status=400)
            return

        if length <= 0 or length > MAX_QUESTION_CHARS * 4:
            self.send_json({"error": "Question missing or too long."}, status=400)
            return

        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            question = str(request["question"]).strip()
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            self.send_json({"error": "Expected JSON with a 'question' field."},
                           status=400)
            return

        if not question:
            self.send_json({"error": "That question was empty."}, status=400)
            return

        print(f"  asked: {question}")
        self.send_json(answer_question(question[:MAX_QUESTION_CHARS]))

    def handle_listen(self) -> None:
        """Transcribe an uploaded recording from the browser's microphone."""
        audio = self.read_body(MAX_AUDIO_BYTES)
        if not audio:
            self.send_json({"error": "No audio received."}, status=400)
            return

        result = transcribe_upload(audio, self.headers.get("Content-Type", ""))
        print(f"  heard: {result.get('text', result.get('error'))}")
        self.send_json(result)

    def handle_speak(self) -> None:
        """Read an answer aloud, once the board has finished writing it."""
        body = self.read_body(MAX_QUESTION_CHARS * 20)
        if not body:
            self.send_json({"error": "Nothing to speak."}, status=400)
            return

        try:
            text = str(json.loads(body.decode("utf-8"))["text"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            self.send_json({"error": "Expected JSON with a 'text' field."},
                           status=400)
            return

        speak_in_background(text)
        self.send_json({"speaking": True}, status=202)

    def log_message(self, format: str, *args) -> None:
        """Quieten the default per-request logging; do_POST prints what matters."""
        return


def main():
    if len(sys.argv) > 2:
        print('Usage: python board_server.py ["<path_to_pdf>"]')
        sys.exit(1)

    if not BOARD_PATH.exists():
        print(f"Can't find {BOARD_PATH.name} next to this script.")
        sys.exit(1)

    if len(sys.argv) == 2:
        pdf_path = sys.argv[1]
        try:
            pages = extract_text_from_pdf(pdf_path)
        except FileNotFoundError as error:
            print(error)
            sys.exit(1)

        if not pages:
            print("No extractable text found. This file might be scanned/image-only")
            print("— it'll need OCR, which isn't built yet.")
            sys.exit(1)

        chunks = chunk_pages(pages, source=slugify_source(pdf_path))
        print(f"Read {len(pages)} page(s), split into {len(chunks)} chunk(s).")
        store_chunks(chunks)

    stored = get_collection().count()
    print(f"Collection holds {stored} chunk(s).")
    if stored == 0:
        print("Nothing stored yet — pass a PDF path, or run store.py first.")

    server = ThreadingHTTPServer((HOST, PORT), BoardHandler)
    print(f"\nBoard ready at http://{HOST}:{PORT}   (Ctrl-C to stop)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
