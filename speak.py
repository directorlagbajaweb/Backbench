"""
Step 5 of Backbench's RAG core: read the answer out loud.

Usage:
    python speak.py "fifty divided by two hundred equals nought point two five"

Voice output only — the bot speaks what it has already printed. Listening to the
student speak is a separate step and isn't built yet.

This uses whatever text-to-speech the operating system already has, starting with
macOS's built-in `say`. That's a deliberate choice over a cloud TTS API: it costs
nothing, needs no key, works offline, and — unlike the Gemini free tier the
teaching step runs on — has no daily request quota to exhaust while testing.

Speech is strictly additive. speak() never raises: no audio device, no TTS
binary, an unsupported platform, or a crash mid-sentence all end the same way,
with the text answer already printed and the session carrying on. Set
BACKBENCH_VOICE=off to silence it without touching the calling code.

On symbols. macOS `say` was measured rather than assumed, by rendering pairs of
phrases to audio and comparing durations. It already ignores markdown — `**`,
`#`, backticks and bullet markers are silent — and already pronounces `=` as
"equals" and `%` as "percent". But it says nothing at all for `/`, so "50 / 200 =
0.25" is heard as "50 200 equals 0.25", which is worse than useless in an answer
whose whole point is correcting arithmetic. So the markup is stripped and the
maths is spelled out in words: the stripping keeps other backends (which do read
punctuation aloud) sounding right, and spelling out the maths fixes the division.
"""

import os
import platform
import re
import shutil
import subprocess
import sys

# Resolved once by find_backend(), then reused. False means "looked and found
# nothing", which is different from None meaning "haven't looked yet".
_backend = None
_warned = False


def to_spoken_text(text: str) -> str:
    """
    Turn a printed answer into something that reads correctly aloud.

    Returns a string like:
        "The actual math: 50 divided by 200 equals 0.25, which is 25 percent"

    Strips markdown to plain prose, then spells out the arithmetic operators, so
    the same text sounds right whichever backend ends up speaking it.
    """
    spoken = text

    # Code fences and inline code: keep the contents, drop the markers.
    spoken = re.sub(r"```[\w-]*\n?", " ", spoken)
    spoken = spoken.replace("`", "")

    # [label](url) -> label. Nobody wants a URL read out character by character.
    spoken = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", spoken)

    # Bold and italic markers, then any stray asterisks left behind.
    spoken = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", spoken)
    spoken = re.sub(r"__([^_]+)__", r"\1", spoken)
    spoken = spoken.replace("*", "")

    # Headings, bullet markers and horizontal rules, line by line. Newlines are
    # left in place: every backend reads them as a pause, which is roughly the
    # rhythm the bullets were providing.
    spoken = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", spoken)
    spoken = re.sub(r"(?m)^\s*([-*_]\s*){3,}$", "", spoken)
    spoken = re.sub(r"(?m)^\s*[-*+]\s+", "", spoken)

    # Maths. `/` is the load-bearing one — macOS say is silent on it, so a
    # division would otherwise be heard as two unrelated numbers. Only digits
    # around the slash, to leave file paths and and/or alone.
    spoken = re.sub(r"(\d)\s*/\s*(\d)", r"\1 divided by \2", spoken)
    spoken = re.sub(r"(\d)\s*%", r"\1 percent", spoken)
    spoken = re.sub(r"\s*=\s*", " equals ", spoken)

    spoken = re.sub(r"[ \t]+", " ", spoken)
    spoken = re.sub(r"(?m)^[ \t]+", "", spoken)  # bullets removed above leave one behind
    spoken = re.sub(r"\n{3,}", "\n\n", spoken)

    return spoken.strip()


def find_backend() -> list[str] | None:
    """
    Work out how to speak on this machine, once.

    Returns a command prefix like ["say"], or None if nothing is available.

    macOS first, since `say` ships with the OS and needs no install. The Linux
    options are the usual suspects; Windows has no branch here rather than
    untested quoting, so it falls through to the graceful no-op.
    """
    if platform.system() == "Darwin" and shutil.which("say"):
        return ["say"]

    for command in ("espeak-ng", "espeak"):
        if shutil.which(command):
            return [command]

    if shutil.which("spd-say"):
        return ["spd-say", "--wait"]

    return None


def get_backend() -> list[str] | None:
    """
    Return the cached backend, probing the system the first time only.

    Worth caching: this runs before every answer, and shutil.which() walks PATH.
    """
    global _backend

    if _backend is None:
        _backend = find_backend() or False

    return _backend or None


def warn_once(message: str) -> None:
    """
    Explain the silence the first time, then stay quiet about it.

    An answer-by-answer complaint about a missing audio device would be noisier
    than the missing audio.
    """
    global _warned

    if not _warned:
        _warned = True
        print(f"(voice off: {message})")


def speak(text: str) -> None:
    """
    Say text out loud, and never let that failing matter.

    Returns nothing and raises nothing — the caller has already printed the
    answer, so speech is a bonus and any problem with it is worth a single note
    rather than an interruption.

    Blocks until the speech finishes, so answers don't talk over each other.
    Ctrl-C during playback stops the speaking and returns to the prompt instead of
    ending the session; press it again at the prompt to quit.
    """
    if os.environ.get("BACKBENCH_VOICE", "").lower() in {"off", "0", "false", "no"}:
        return

    if not isinstance(text, str) or not text.strip():
        return

    # Everything below is inside the try, including the text munging and the
    # backend probe, so the promise in this docstring holds for the whole body
    # and not just the part that touches the audio device.
    process = None
    try:
        spoken = to_spoken_text(text)
        if not spoken:
            return

        backend = get_backend()
        if backend is None:
            warn_once("no text-to-speech command found on this system")
            return

        process = subprocess.Popen(backend + [spoken])
        process.wait()
    except KeyboardInterrupt:
        # Stop talking, keep the session. Swallowed on purpose: re-raising here
        # would tear down a loop the user only wanted to quieten.
        if process is not None:
            process.terminate()
        print()
    except Exception as error:
        warn_once(f"{type(error).__name__}: {error}")


def main():
    if len(sys.argv) != 2:
        print('Usage: python speak.py "<text to read aloud>"')
        sys.exit(1)

    backend = get_backend()
    print(f"backend: {' '.join(backend) if backend else 'none available'}")
    print(f"text: {to_spoken_text(sys.argv[1])!r}")

    speak(sys.argv[1])


if __name__ == "__main__":
    main()
