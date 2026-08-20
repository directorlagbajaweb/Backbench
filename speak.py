"""
Step 5 of Backbench's RAG core: read the answer out loud.

Usage:
    python speak.py "fifty divided by two hundred equals nought point two five"

Voice output only — the bot speaks what it has already printed. Listening to the
student speak lives in listen.py.

Two backends, in order. ElevenLabs is the primary voice: a natural-sounding
neural TTS over plain HTTP, which is a large step up from the built-in system
voice. The operating system's own text-to-speech is kept as an automatic
fallback, not replaced — no key, a failed request, an exhausted free quota, or
no way to play an MP3 all drop straight back to `say` with a single note. That
matters because voice is additive to a conversation: it should degrade, never
interrupt.

Text normalisation runs once, before either backend. ElevenLabs has no more
business reading "**" aloud or silently swallowing "/" than `say` does.

Speech is strictly additive. speak() never raises. Set BACKBENCH_VOICE=off to
silence it entirely, or BACKBENCH_VOICE=say to force the fallback backend — handy
for hearing the difference between the two without touching your key.

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
import tempfile

import httpx

# --- ElevenLabs -------------------------------------------------------------
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVEN_TIMEOUT = 60

# Chosen from ElevenLabs' own premade voice list (GET /v1/voices, which answers
# unauthenticated), not from memory. Of the 21 premade voices, this one is
# labelled descriptive="calm" and named "Relaxed, Neutral, Informative" — the
# closest thing in the library to a calm, clear tutor.
#
# It also matters that River lists eleven_multilingual_v2 among its supported
# models. Several otherwise-tempting voices don't: "Alice - Clear, Engaging
# Educator" is the most on-the-nose name in the catalogue but supports only the
# flash/turbo models, so pairing it with the higher-quality model would have been
# a guess.
#
# Documented alternatives, all premade, all use_case=informative_educational:
#   hpp4J3VqNfWAUOO0d1Us  Bella  professional, bright, warm (also multilingual_v2)
#   Xb7hH8MSUJpSbSDYk0k2  Alice  clear, engaging educator (flash models only)
#   onwK4e9ZLuTAKqWW03F9  Daniel steady broadcaster, formal, british
ELEVEN_VOICE_ID = "SAz9YHcvj6GT2YYXdXww"
ELEVEN_VOICE_NAME = "River"

# eleven_multilingual_v2 over the flash models deliberately. ElevenLabs describes
# it as lifelike with consistent quality and stable long-form generation, against
# flash's ~75ms-latency-first tradeoff. Latency barely matters here: the answer is
# already written on screen before a word is spoken, so quality wins. (The turbo
# models are deprecated in favour of flash — not used.)
ELEVEN_MODEL = "eleven_multilingual_v2"

# Slightly above the middle on stability, which trades expressive variation for a
# steady, even delivery — the right side of that tradeoff for explaining things.
ELEVEN_VOICE_SETTINGS = {"stability": 0.55, "similarity_boost": 0.75}

# Players for the MP3 that comes back. afplay ships with macOS, so on this
# machine there's nothing to install.
AUDIO_PLAYERS = (
    ["afplay"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpg123", "-q"],
)

# Outcomes of trying to speak, so a deliberate Ctrl-C is never mistaken for a
# failure. Falling back after a cancel would start the whole answer again in a
# different voice, which is the opposite of what stopping it meant.
SPOKE, FAILED, CANCELLED = "spoke", "failed", "cancelled"

# Resolved once, then reused. False means "looked and found nothing", which is
# different from None meaning "haven't looked yet".
_backend = None
_player = None
_eleven_key = None
_warned = set()


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
    Work out how to speak with the operating system's own voice, once.

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
    Return the cached system-voice backend, probing the system the first time only.

    Worth caching: this runs before every answer, and shutil.which() walks PATH.
    """
    global _backend

    if _backend is None:
        _backend = find_backend() or False

    return _backend or None


def get_player() -> list[str] | None:
    """Return the cached command for playing an MP3, or None if there isn't one."""
    global _player

    if _player is None:
        _player = next(
            (p for p in AUDIO_PLAYERS if shutil.which(p[0])), False
        )

    return _player or None


def get_eleven_key() -> str | None:
    """
    Return the ElevenLabs key from .env, or None if it isn't set.

    A missing key is an ordinary, expected state here rather than an error: it
    just means the fallback voice is the one that speaks.
    """
    global _eleven_key

    if _eleven_key is None:
        from dotenv import load_dotenv

        load_dotenv()
        _eleven_key = os.getenv("ELEVENLABS_API_KEY") or False

    return _eleven_key or None


def warn_once(message: str) -> None:
    """
    Explain a fallback or a silence the first time, then stay quiet about it.

    Keyed on the message rather than a single flag, so a missing key and a later
    quota failure each get one line — but neither repeats on every answer.
    """
    if message not in _warned:
        _warned.add(message)
        print(f"(voice: {message})")


def synthesise(spoken: str) -> bytes | None:
    """
    Ask ElevenLabs for audio of already-normalised text.

    Returns MP3 bytes, or None if the request couldn't be made or was refused —
    in which case the caller falls back to the system voice. Every failure mode
    is a note, never an exception: a quota that ran out mid-conversation should
    change how the answer sounds, nothing more.
    """
    key = get_eleven_key()
    if not key:
        warn_once("no ELEVENLABS_API_KEY set, using the system voice")
        return None

    try:
        response = httpx.post(
            f"{ELEVEN_URL}/{ELEVEN_VOICE_ID}",
            headers={"xi-api-key": key, "Accept": "audio/mpeg"},
            json={
                "text": spoken,
                "model_id": ELEVEN_MODEL,
                "voice_settings": ELEVEN_VOICE_SETTINGS,
            },
            timeout=ELEVEN_TIMEOUT,
        )
    except httpx.HTTPError as error:
        warn_once(f"couldn't reach ElevenLabs ({type(error).__name__}), "
                  f"using the system voice")
        return None

    if response.status_code == 200:
        return response.content

    # 401 covers both a bad key and, on the free tier, an exhausted quota; the
    # body says which. Keep it to one line rather than dumping the JSON.
    detail = ""
    try:
        body = response.json()
        detail = str(body.get("detail") or body)[:160]
    except ValueError:
        detail = (response.text or "")[:160]

    warn_once(f"ElevenLabs returned {response.status_code} ({detail}), "
              f"using the system voice")
    return None


def play_audio(data: bytes) -> str:
    """
    Play MP3 bytes, blocking until they finish.

    Returns SPOKE, FAILED or CANCELLED. FAILED means the caller should try the
    system voice; CANCELLED means the listener stopped it deliberately and
    nothing else should start talking.
    """
    player = get_player()
    if player is None:
        warn_once("no MP3 player found, using the system voice")
        return FAILED

    path = None
    process = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            handle.write(data)
            path = handle.name

        process = subprocess.Popen(player + [path],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        return SPOKE if process.wait() == 0 else FAILED
    except KeyboardInterrupt:
        if process is not None:
            process.terminate()
        print()
        return CANCELLED
    except Exception as error:
        warn_once(f"couldn't play the audio ({type(error).__name__}), "
                  f"using the system voice")
        return FAILED
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def speak_with_system(spoken: str) -> str:
    """
    Speak already-normalised text with the operating system's own voice.

    Returns SPOKE, FAILED or CANCELLED. This is the fallback path, and it is the
    same code that used to be the only path.
    """
    backend = get_backend()
    if backend is None:
        warn_once("no text-to-speech command found on this system")
        return FAILED

    process = None
    try:
        process = subprocess.Popen(backend + [spoken])
        process.wait()
        return SPOKE
    except KeyboardInterrupt:
        # Stop talking, keep the session. Swallowed on purpose: re-raising here
        # would tear down a loop the user only wanted to quieten.
        if process is not None:
            process.terminate()
        print()
        return CANCELLED
    except Exception as error:
        warn_once(f"{type(error).__name__}: {error}")
        return FAILED


def speak(text: str) -> None:
    """
    Say text out loud, and never let that failing matter.

    Returns nothing and raises nothing — the caller has already printed the
    answer, so speech is a bonus and any problem with it is worth a single note
    rather than an interruption.

    Tries ElevenLabs first and falls back to the system voice on any failure.
    Blocks until the speech finishes, so answers don't talk over each other.
    Ctrl-C during playback stops the speaking and returns to the prompt instead
    of ending the session — and does not then restart the answer in the fallback
    voice; press it again at the prompt to quit.
    """
    setting = os.environ.get("BACKBENCH_VOICE", "").lower()
    if setting in {"off", "0", "false", "no"}:
        return

    if not isinstance(text, str) or not text.strip():
        return

    # Everything below is inside the try, including the text munging and the
    # backend probes, so the promise in this docstring holds for the whole body
    # and not just the parts that touch an audio device.
    try:
        spoken = to_spoken_text(text)
        if not spoken:
            return

        # BACKBENCH_VOICE=say skips ElevenLabs entirely, which is the easiest way
        # to hear the two voices back to back.
        if setting not in {"say", "system", "fallback"}:
            audio = synthesise(spoken)
            if audio:
                result = play_audio(audio)
                if result in (SPOKE, CANCELLED):
                    return

        speak_with_system(spoken)
    except Exception as error:
        warn_once(f"{type(error).__name__}: {error}")


def main():
    if len(sys.argv) != 2:
        print('Usage: python speak.py "<text to read aloud>"')
        sys.exit(1)

    forced = os.environ.get("BACKBENCH_VOICE", "").lower() in {
        "say", "system", "fallback"}
    print(f"primary  : ElevenLabs {ELEVEN_VOICE_NAME} / {ELEVEN_MODEL}"
          f"{' (skipped: BACKBENCH_VOICE)' if forced else ''}")
    print(f"  key set: {bool(get_eleven_key())}   mp3 player: "
          f"{(get_player() or ['none'])[0]}")
    backend = get_backend()
    print(f"fallback : {' '.join(backend) if backend else 'none available'}")
    print(f"text     : {to_spoken_text(sys.argv[1])!r}")

    speak(sys.argv[1])


if __name__ == "__main__":
    main()
