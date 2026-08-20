"""
Step 3 of Backbench's RAG core: teach from the retrieved chunks, and only those.

Usage:
    python teach.py "what research design did the study use?"

This needs an OpenRouter API key. Add OPENROUTER_API_KEY to your .env file.

Provider note: this runs on OpenRouter, which fronts many providers behind one
OpenAI-compatible endpoint. It replaced Gemini because Gemini's free tier allows
only 20 requests per day per model, which is spent within an afternoon of
testing. Before that it ran on Claude. Everything provider-specific stays
confined to the same three places it always has — MODEL, get_client(), and the
request/response handling inside generate_answer(). The system prompt, the chunk
and history formatting, the CLI, and generate_answer's signature are all
provider-agnostic, and anthropic and google-genai stay pinned in
requirements.txt, so switching back or comparing is an edit to those three spots.

No SDK for this one. OpenRouter speaks OpenAI-compatible JSON over plain HTTP,
which is one POST — so this uses httpx, already present, rather than adding the
openai package for a single call.

The whole job here is grounding: take a question plus the chunks store.py
retrieved for it, and produce an answer that teaches from those chunks and
nothing else. Two findings from testing Step 2 on real material shape how this
works.

First, distance scores don't separate "wrong answer" from "no answer" cleanly —
an unrelated question scored 0.988 while a real question that retrieval answered
badly scored 0.826, close enough that a simple distance threshold would be
unreliable. So distances are passed to the model as context to reason about, not
used as a gate that filters chunks out before it ever sees them.

Second, grounding has to be instructed, not assumed. The system prompt says
explicitly that saying "your material doesn't cover this" is the right answer
even when the model privately knows better.

Deferred: conversation history (each question is answered fresh), and wiring
this into main.py's loop.
"""

import os
import re
import sys

import httpx
from dotenv import load_dotenv

from store import search_chunks

# Picked from OpenRouter's live model list (their /api/v1/models endpoint), taking
# only entries whose prompt *and* completion price are both zero — 20 of 417
# models at the time of writing.
#
# This one is the frontier-reasoning option among them: a 550B mixture-of-experts
# with 55B active and a 1M-token context. The choice is deliberate rather than
# first-on-the-list, because this prompt leans hard on judgement — deciding
# whether retrieved excerpts genuinely answer the question, declining when they
# don't, and spotting arithmetic that contradicts itself. Small fast models tend
# to be agreeable instead of disciplined about exactly that.
#
# Documented alternatives, all free:
#   z-ai/glm-5.2:free                      reasoning-focused, 256K, likely faster
#   nvidia/nemotron-3-super-120b-a12b:free 120B/12B active, efficiency-minded
#   google/gemma-4-31b-it:free             31B dense, instruction-tuned
#
# Deliberately NOT openrouter/free: it picks a free model at random per request,
# so grounding behaviour would vary run to run and a carefully tuned prompt could
# not be held responsible for the output.
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0.2  # grounded teaching, not creative writing — keep it close to the source
MAX_TOKENS = 2000
TIMEOUT_SECONDS = 180  # free-tier reasoning models can be slow to start
N_RESULTS = 3
RULE = "=" * 70

# finish_reason values that mean "no usable answer", as opposed to simply running
# long. Checked explicitly because they arrive as a perfectly successful HTTP 200.
BLOCKED_FINISH_REASONS = frozenset({"content_filter", "error"})

# Reasoning models on OpenRouter sometimes leak their scratchpad into the answer
# as <think>…</think>. Left in, it would be printed on the board and read aloud.
THINK_TAGS = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)

SYSTEM_PROMPT = """\
You are Backbench, a teaching bot that helps a student learn from their own \
course material — their lecture notes, textbook pages, and handouts.

The excerpts in the user's message are the entire body of knowledge available to \
you for answering. Each one carries the page it came from and a distance score \
from the search that found it, where a lower distance means a closer match.

# Answer only from the excerpts

Teach only what the excerpts support. Do not add facts, definitions, examples, \
dates, names, or figures that aren't in them, even when you're confident you \
know the answer from general knowledge. The student is studying for a specific \
course and is being examined on what their material says.

When the excerpts don't answer the question, say so plainly — "your material \
doesn't cover this" — and then say what they do cover, so the student can \
rephrase or go looking elsewhere. That is a useful answer, not a failure.

Say that even when you privately know the answer. If you quietly fill the gap \
from general knowledge, the student has no way to tell which parts of your \
answer came from their course and which came from you — and that's exactly the \
distinction they need at exam time.

# Judge relevance yourself

The excerpts were chosen by a similarity search that always returns its closest \
matches, however far away those are. It has no notion of "no result", so being \
handed an excerpt is not evidence that the excerpt is relevant.

The distance scores are a weak signal, not a threshold. In testing, a question \
the material genuinely didn't cover and a question retrieval simply answered \
badly landed close enough together that no cut-off separated them. Read the \
scores, but don't defer to them.

So read the excerpts and decide for yourself whether they actually answer the \
question that was asked. The most common failure is an excerpt on the same broad \
topic that doesn't address the specific question — a methodology section \
retrieved for a question about conclusions, say. Treat that as not covered. If \
only part of the question is covered, answer that part and be explicit about \
which part isn't.

# Teach, don't quote

Explain the idea in your own words, in the order that makes it easiest to \
follow — the way a lecturer talks a student through a concept, building up from \
what the material establishes first. Quote a phrase verbatim only when the exact \
wording carries weight, such as a formal definition.

Cite the page number when you draw on a specific excerpt, so the student can go \
and read the original.

Match the length to the question. A short factual question gets a short answer; \
save the step-by-step build-up for ideas that genuinely need it.

Write mathematics as plain text: no LaTeX, no $ delimiters, no \\frac{}{}. Use / \
for division and ^ for exponents — the student reads your answers in a terminal, \
where LaTeX markup is unreadable noise.

# Follow-up questions

The user's message may open with earlier questions and answers from this \
conversation. That history is there for one purpose: working out what a short \
follow-up actually means. "Why?" after an answer about sampling means "why was \
that sampling method chosen?", and "can you explain that more?" refers to what \
you just explained, not to the words in isolation.

Use it for that and nothing else. It is not a source of facts. The excerpts are \
still the only material you may teach from, and something you asserted earlier \
does not become citable just because you were the one who said it. You may refer \
back to what you already told them — recalling your own explanation is not the \
same as inventing material — but any fact you add must come from the excerpts in \
front of you now.

So if the excerpts don't support the follow-up, say so plainly, exactly as you \
would for a fresh question. A follow-up is not a licence to keep talking past \
where the material runs out.

# Catch mistakes in the material

Course material is sometimes wrong, and teaching an error as fact is worse than \
not teaching at all. If an excerpt contains a clear factual, mathematical, or \
logical error — arithmetic that doesn't add up, a worked example that reaches a \
conclusion its own steps don't support, a claim that contradicts something else \
in the material — say plainly that it looks wrong, show precisely what the error \
is, and then teach the corrected version.

Only do this for errors you can demonstrate from the excerpt itself. Don't \
second-guess phrasing you merely find unusual, and don't label a claim an error \
just because it's outside what the excerpts establish.\
"""

# Chroma's client is built lazily elsewhere for the same reason: one per process.
_client = None


def extract_error(response) -> str:
    """
    Pull a readable message out of a failed response.

    OpenRouter normally returns {"error": {"message": ...}}, but a proxy or a
    gateway in front of it may return HTML. Falls back to a trimmed body rather
    than dumping a whole error page into the terminal.
    """
    try:
        body = response.json()
        if isinstance(body.get("error"), dict):
            return str(body["error"].get("message") or body["error"])
        return str(body)[:300]
    except ValueError:
        return (response.text or "").strip()[:300] or "no detail returned"


class OpenRouterError(Exception):
    """
    An API failure, carrying .code and .message.

    Those two attribute names are not arbitrary. main.py and board_server.py
    already read them by duck-typing rather than importing any provider's
    exception classes, so raising this keeps both callers working with no edit —
    the same reason the Gemini errors it replaces needed none.
    """

    def __init__(self, code: int | None, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}" if code else message)


def get_client():
    """
    Build (once) the HTTP client for OpenRouter, loading the key from .env.

    Raises RuntimeError with a fix-it message if no key is configured — that's
    the most likely first-run failure, and a bare 401 doesn't tell you which
    variable to go and set.
    """
    global _client

    if _client is None:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No OPENROUTER_API_KEY found. Add this line to your .env file:\n"
                "    OPENROUTER_API_KEY=sk-or-v1-your_key_here\n"
                "Free keys come from https://openrouter.ai/keys"
            )
        _client = httpx.Client(
            timeout=TIMEOUT_SECONDS,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # Optional attribution headers OpenRouter uses to label traffic.
                "X-Title": "Backbench",
            },
        )

    return _client


def format_chunks(chunks: list[dict]) -> str:
    """
    Lay the retrieved chunks out for the prompt, metadata and all.

    Returns a string like:
        [Excerpt 1] page 4 — distance 0.731 (lower is closer)
        Thermodynamics is the study of heat ...

    The page number is what lets an answer cite itself. The distance is included
    on purpose rather than stripped: the system prompt asks the model to weigh
    relevance itself, and it can only do that if it can see how far away each
    excerpt actually was.
    """
    blocks = []

    for rank, chunk in enumerate(chunks, start=1):
        header = f"[Excerpt {rank}] page {chunk.get('page', '?')}"
        distance = chunk.get("distance")
        if distance is not None:
            header += f" — distance {distance:.3f} (lower is closer)"
        blocks.append(f"{header}\n{chunk.get('text', '').strip()}")

    return "\n\n".join(blocks)


def format_history(history: list[dict]) -> str:
    """
    Lay out earlier exchanges so a short follow-up can be understood.

    Returns a string like:
        Earlier in this conversation:

        Student asked: What research design did the study use?
        You answered: This study used a descriptive survey design...

    Labelled as conversation rather than as material, deliberately: the system
    prompt tells the model these are not citable facts, and the wording here
    shouldn't undercut that by making them look like more excerpts.
    """
    turns = []

    for exchange in history:
        turns.append(
            f"Student asked: {exchange.get('question', '')}\n"
            f"You answered: {exchange.get('answer', '')}"
        )

    return "Earlier in this conversation:\n\n" + "\n\n".join(turns)


def build_user_message(question: str, chunks: list[dict],
                       history: list[dict] | None = None) -> str:
    """
    Assemble the single user turn: any history, then the excerpts, then the question.

    Returns a string like:
        Earlier in this conversation:
        Student asked: ...
        You answered: ...
        ---
        Excerpts retrieved from the student's course material:
        [Excerpt 1] page 4 — distance 0.731 (lower is closer)
        ---
        The student's question: why?

    The question goes last so it reads as the thing being asked *about* the
    material above it, rather than getting lost in front of a wall of excerpts.
    History goes first, as background, so the excerpts stay next to the question
    they're supposed to answer.
    """
    parts = []

    if history:
        parts.append(format_history(history))

    parts.append(
        "Excerpts retrieved from the student's course material:\n\n"
        f"{format_chunks(chunks)}"
    )
    parts.append(f"The student's question: {question}")

    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict],
                    history: list[dict] | None = None) -> str:
    """
    Ask the model to teach the answer to one question from one set of chunks.

    Takes the question, the chunks that search_chunks() returned for it, and
    optionally the recent exchanges of this conversation as
    [{"question": ..., "answer": ...}, ...] — oldest first. Gives back the answer
    as plain text.

    History is passed so short follow-ups can be understood, not so they can be
    answered from it: the excerpts remain the only source of facts, and the
    system prompt says so explicitly. Callers that want no memory — the board
    server, for one — simply leave it out and get the old behaviour.

    Note the split of responsibilities: the history lives with the caller, so
    this stays a pure function of what it's handed and nothing here accumulates
    state between questions.
    """
    if not chunks:
        return (
            "Nothing came back from your material for that question, so there's "
            "nothing for me to teach from. Is anything actually stored?"
        )

    # The system prompt is a separate message here rather than a dedicated
    # parameter, which is the only structural difference from the Gemini call.
    # Its text is untouched.
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, chunks, history)},
        ],
    }

    try:
        response = get_client().post(API_URL, json=payload)
    except httpx.TimeoutException:
        raise OpenRouterError(None, f"No response within {TIMEOUT_SECONDS}s. "
                                    f"Free models can be slow — try again.")
    except httpx.HTTPError as error:
        raise OpenRouterError(None, f"Couldn't reach OpenRouter: {error}")

    if response.status_code != 200:
        raise OpenRouterError(response.status_code, extract_error(response))

    try:
        body = response.json()
    except ValueError:
        raise OpenRouterError(None, "OpenRouter returned something that wasn't JSON.")

    # OpenRouter can also report failure inside a 200 body.
    if isinstance(body.get("error"), dict):
        raise OpenRouterError(body["error"].get("code"),
                              str(body["error"].get("message", "Unknown error")))

    choices = body.get("choices") or []
    if not choices:
        return "OpenRouter returned no answer at all, which shouldn't happen."

    choice = choices[0]
    finish_reason = choice.get("finish_reason")

    if finish_reason in BLOCKED_FINISH_REASONS:
        return (
            f"The model stopped without answering ({finish_reason}). "
            "A content filter flagged the material or the answer — try "
            "rephrasing the question."
        )

    answer = THINK_TAGS.sub("", choice.get("message", {}).get("content") or "").strip()

    if finish_reason == "length":
        answer += f"\n\n[Answer was cut off at the {MAX_TOKENS}-token limit.]"

    return answer or "OpenRouter returned an empty answer, which shouldn't happen."


def main():
    if len(sys.argv) != 2:
        print('Usage: python teach.py "<your question>"')
        sys.exit(1)

    question = sys.argv[1].strip()
    if not question:
        print("That's an empty question.")
        sys.exit(1)

    chunks = search_chunks(question, n_results=N_RESULTS)

    # Print what went in before what came out. If an answer looks wrong, the
    # first thing you need to know is whether retrieval or teaching went wrong.
    print(f"Retrieved {len(chunks)} chunk(s) for: {question}")
    print("(distance = how far a chunk sits from the question — lower is closer)\n")
    for rank, chunk in enumerate(chunks, start=1):
        preview = chunk["text"][:160].replace("\n", " ")
        print(f"  [{rank}] page {chunk['page']}  {chunk['chunk_id']}  "
              f"distance {chunk['distance']:.3f}")
        print(f"      {preview}...")

    print(f"\n{RULE}")
    try:
        print(generate_answer(question, chunks))
    except RuntimeError as error:
        print(error)
        sys.exit(1)
    except OpenRouterError as error:
        if error.code == 429:
            # Free models are rate limited per model and per account. The API's
            # own message says which limit was hit, so pass it through rather
            # than guessing.
            print(f"Rate limited: {error.message}")
            print("Free models have their own limits, counted per model — so")
            print(f"changing MODEL (currently {MODEL}) is the usual fix. The")
            print("alternatives are listed in the comment above it.")
        elif error.code in (401, 403):
            print(f"OpenRouter rejected the key ({error.code}): {error.message}")
            print("Check OPENROUTER_API_KEY in .env — keys start with 'sk-or-v1-'.")
        elif error.code == 404:
            print(f"No such model: {MODEL}")
            print("Free models come and go. Check https://openrouter.ai/models")
            print("and pick another, filtering for free.")
        elif error.code and error.code >= 500:
            print(f"OpenRouter server error {error.code}: {error.message}")
            print("Not your fault — try again shortly.")
        else:
            print(f"OpenRouter error: {error.message}")
        sys.exit(1)
    print(RULE)


if __name__ == "__main__":
    main()
