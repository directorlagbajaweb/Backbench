"""
Step 3 of Backbench's RAG core: teach from the retrieved chunks, and only those.

Usage:
    python teach.py "what research design did the study use?"

This needs a Gemini API key. Add GEMINI_API_KEY to your .env file.

Provider note: this runs on Google's Gemini API rather than Claude for now,
because Gemini's free tier works without billing set up. Everything provider-
specific is deliberately confined to three places — MODEL, get_client(), and the
request/response handling inside generate_answer(). The system prompt, the chunk
formatting, the CLI, and generate_answer's signature are all provider-agnostic,
so switching back to Claude means editing those three spots and nothing else.

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
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from store import search_chunks

# Free tier. Picked over gemini-3.7-flash on measured reliability: 3.7 is the
# more capable model but returned 503 "high demand" on a third of calls during
# testing, while this one answered every time. Switch to gemini-3.7-flash if it's
# responsive and you want the extra capability. Note gemini-2.5-flash 404s — it
# isn't served on this API version, despite appearing in the pricing tables.
MODEL = "gemini-3.5-flash"
TEMPERATURE = 0.2  # grounded teaching, not creative writing — keep it close to the source
N_RESULTS = 3
RULE = "=" * 70

# Ways the model can stop that mean "there is no usable answer here", as opposed
# to simply running long. Checked explicitly because every one of them arrives as
# a perfectly successful HTTP response.
BLOCKED_FINISH_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.SPII,
        types.FinishReason.RECITATION,
    }
)

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


def get_client():
    """
    Build (once) the Gemini client, loading the key from .env.

    Raises RuntimeError with a fix-it message if no key is configured — that's
    the most likely first-run failure, and a bare 400 from deep inside the SDK
    doesn't tell you which variable to go and set.
    """
    global _client

    if _client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No GEMINI_API_KEY found. Add this line to your .env file:\n"
                "    GEMINI_API_KEY=your_key_here\n"
                "Free keys come from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=api_key)

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


def build_user_message(question: str, chunks: list[dict]) -> str:
    """
    Assemble the single user turn: the excerpts, then the question.

    Returns a string like:
        Excerpts retrieved from the student's course material:
        [Excerpt 1] page 4 — distance 0.731 (lower is closer)
        ...
        ---
        The student's question: What research design did the study use?

    The question goes last so it reads as the thing being asked *about* the
    material above it, rather than getting lost in front of a wall of excerpts.
    """
    return (
        "Excerpts retrieved from the student's course material:\n\n"
        f"{format_chunks(chunks)}\n\n"
        "---\n\n"
        f"The student's question: {question}"
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Ask the model to teach the answer to one question from one set of chunks.

    Takes the question and the chunks that search_chunks() returned for it, and
    gives back the answer as plain text.

    No conversation history: each question is answered from its own excerpts, so
    the answer can never lean on something established in an earlier turn that
    the current excerpts don't support.
    """
    if not chunks:
        return (
            "Nothing came back from your material for that question, so there's "
            "nothing for me to teach from. Is anything actually stored?"
        )

    response = get_client().models.generate_content(
        model=MODEL,
        contents=build_user_message(question, chunks),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            # No tools here, so automatic function calling has nothing to do —
            # left on, the SDK logs a warning about it on every single run.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    # Gemini can decline in two separate places, and both come back as a
    # successful response rather than an error: it can block the prompt before
    # generating anything, or stop partway through its own answer. This material
    # is a cybersecurity paper, so the safety filters are a live possibility
    # rather than a theoretical one.
    block_reason = getattr(response.prompt_feedback, "block_reason", None)
    if block_reason:
        return (
            f"Gemini blocked the question before answering ({block_reason}). "
            "Its safety filters flagged the request — try rephrasing it."
        )

    if not response.candidates:
        return "Gemini returned no answer at all, which shouldn't happen."

    finish_reason = response.candidates[0].finish_reason
    if finish_reason in BLOCKED_FINISH_REASONS:
        return (
            f"Gemini stopped partway through its answer ({finish_reason.value}). "
            "Its safety filters flagged the material or the answer — try "
            "rephrasing the question."
        )

    # .text is None when the response holds no text parts at all.
    answer = (response.text or "").strip()

    if finish_reason == types.FinishReason.MAX_TOKENS:
        answer += "\n\n[Answer was cut off at the model's output limit.]"

    return answer or "Gemini returned an empty answer, which shouldn't happen."


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
    except errors.ClientError as error:
        if error.code == 429:
            print("Rate limited — the free tier allows only a few requests per")
            print("minute. Wait a moment and try again.")
        elif error.code in (400, 403):
            print(f"Gemini rejected the request ({error.code}): {error.message}")
            print("If that mentions the API key, check GEMINI_API_KEY in .env.")
        else:
            print(f"Gemini client error {error.code}: {error.message}")
        sys.exit(1)
    except errors.ServerError as error:
        print(f"Gemini server error {error.code}: {error.message}")
        print("Not your fault — try again shortly.")
        sys.exit(1)
    except errors.APIError as error:
        print(f"Gemini API error: {error}")
        sys.exit(1)
    print(RULE)


if __name__ == "__main__":
    main()
