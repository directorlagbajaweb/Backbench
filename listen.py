"""
Step 6 of Backbench's RAG core: hear the question as well as read it.

Usage:
    python listen.py

Voice input, sitting beside typing rather than replacing it. main.py still reads
typed questions exactly as before; this is what runs when you ask for it.

Both halves are free and offline. Recording is sounddevice, whose wheel bundles
PortAudio — which matters because there's no Homebrew on this machine, so
anything needing `brew install portaudio` was never an option. Transcription is
faster-whisper running locally on the CPU. Nothing here needs a key, and nothing
here touches the Gemini quota: a spoken question costs exactly the same one
request as a typed one, because only the teaching step calls an API.

Why faster-whisper over the alternatives: macOS ships no recording command at all
(there's afplay but no afrecord, and no ffmpeg or sox here), and vosk's small
model is noticeably less accurate.

On model size. All of tiny, base and small transcribe clean synthesised speech
perfectly, so clean audio tells you nothing. The difference shows up on real
microphone recordings with background noise: given the identical noisy clip,
"base" mangled "What research design did the study use?" into "a research design
did the study use", while "small" recovered the sentence intact. Hence small,
accepting that it is roughly three times slower.

First run downloads ~464 MB of model to ~/.cache/huggingface and takes a few
minutes — a bigger version of the one-off cost Chroma's embedding model has in
Step 2. After that the model loads in under two seconds and transcribes a
question in about a second. Dropping to "base" (141 MB) or "tiny" (75 MB) trades
accuracy for speed.

Worth knowing before blaming the model: background speech near the microphone
hurts accuracy more than model size does. Whisper transcribes whatever it hears,
so a television or a conversation in the room gets mixed into the question.

listen() never raises. No microphone, no permission, no dependency installed, a
silent room, or Ctrl-C part way through all end the same way: a one-line
explanation and None, so the caller can fall back to asking you to type.
"""

import sys

SAMPLE_RATE = 16000     # what whisper wants; this mic supports it directly
CHANNELS = 1
MODEL_SIZE = "small"    # accuracy over speed: "base" misheard "study" in real use
LANGUAGE = "en"         # forced, because short clips are easy to mis-detect
MAX_SECONDS = 60        # a safety cap, in case a recording is left running
SILENT_RMS = 0.001      # below this the microphone effectively heard nothing

_model = None


def get_model():
    """
    Load the transcription model once and keep it.

    Imported inside the function rather than at the top of the file on purpose:
    main.py imports this module at startup, and neither faster-whisper nor
    sounddevice should be able to stop the typed path from working just by being
    absent or slow to import.
    """
    global _model

    if _model is None:
        from faster_whisper import WhisperModel

        print(f"(loading the {MODEL_SIZE} speech model — first run downloads "
              f"~464 MB and takes a few minutes)")
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    return _model


def record_until_enter():
    """
    Record from the default microphone until Enter is pressed.

    Returns a 1-D float32 numpy array of samples, or None if nothing was
    captured.

    Enter rather than a fixed number of seconds, because a fixed window either
    cuts a long question off or leaves you sitting in silence — and rather than
    silence detection, because "stop when I say so" has no threshold to tune and
    nothing to get wrong in a noisy room.
    """
    import numpy as np
    import sounddevice as sd

    frames = []
    captured = 0
    limit = MAX_SECONDS * SAMPLE_RATE

    def callback(indata, frame_count, time_info, status):
        nonlocal captured
        if captured < limit:
            frames.append(indata.copy())
            captured += len(indata)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", callback=callback):
        print(f"Recording — say your question, then press Enter "
              f"(or Ctrl-C to cancel; {MAX_SECONDS}s max).")
        input()

    if not frames:
        return None

    return np.concatenate(frames, axis=0).flatten()


def is_silent(audio) -> bool:
    """
    Decide whether a recording is effectively empty.

    Worth checking separately from "did it fail": on macOS a microphone the
    terminal lacks permission for records perfectly happily and returns nothing
    but zeros, so without this the failure looks like whisper transcribing
    silence into an empty question.
    """
    import numpy as np

    return float(np.sqrt(np.mean(audio ** 2))) < SILENT_RMS


def transcribe(audio) -> str | None:
    """
    Turn recorded samples into text.

    Returns the transcript, or None if the model found no speech in the audio.
    """
    segments, _ = get_model().transcribe(audio, language=LANGUAGE, beam_size=1)
    text = " ".join(segment.text for segment in segments).strip()

    return text or None


def listen() -> str | None:
    """
    Record a spoken question and return it as text.

    Returns the transcript, or None if anything at all went wrong — a missing
    dependency, no microphone, a denied permission, a silent room, no speech
    found, or a cancelled recording. Every one of those prints a single line
    saying what happened, because the caller's fallback is "type it instead" and
    that's only useful if you know why.
    """
    try:
        audio = record_until_enter()

        if audio is None or len(audio) < SAMPLE_RATE // 10:
            print("(heard nothing — recording was too short. Type it instead.)")
            return None

        if is_silent(audio):
            print("(microphone recorded silence — check it isn't muted, and that "
                  "your terminal has microphone permission in System Settings > "
                  "Privacy & Security. Type it instead.)")
            return None

        text = transcribe(audio)
        if text is None:
            print("(no speech found in that recording. Type it instead.)")
            return None

        return text

    except KeyboardInterrupt:
        print("\n(voice input cancelled. Type it instead.)")
        return None
    except ImportError as error:
        print(f"(voice input needs a package that isn't installed: {error}. "
              f"Type it instead.)")
        return None
    except Exception as error:
        print(f"(voice input failed — {type(error).__name__}: {error}. "
              f"Type it instead.)")
        return None


def main():
    if len(sys.argv) != 1:
        print("Usage: python listen.py")
        sys.exit(1)

    try:
        import sounddevice as sd
        print(f"microphone: {sd.query_devices(kind='input')['name']}")
    except Exception as error:
        print(f"no usable microphone: {type(error).__name__}: {error}")

    heard = listen()
    print(f"\nheard: {heard!r}" if heard else "\nnothing transcribed.")


if __name__ == "__main__":
    main()
