"""
Test fixture builder: a PDF with a deliberate error, for checking that teach.py
catches mistakes instead of repeating them.

Usage:
    python generate_test_pdf.py

The real seminar paper in test_material/ is correct throughout, which means it
can't exercise the one behaviour in teach.py's system prompt that nothing else
tests: spotting a wrong worked example and teaching the corrected version rather
than passing the error on as fact.

So this writes a short page that reads like an excerpt from a paper and contains
one planted arithmetic error — 50 out of 200 firms reported as 15% when it is
actually 25%. The error is stated twice over, once as the percentage and once as
the "fewer than one in six" inference drawn from it, so a model that swallows the
mistake repeats it visibly rather than in a way you might skim past.

The page is deliberately kept under chunk.py's 175-word limit. The numbers and
the false conclusion have to land in the *same* chunk — split across two, the
model could be handed the conclusion without the arithmetic that disproves it,
and then failing to catch the error would be the correct behaviour rather than a
bug.
"""

import sys
from pathlib import Path

import pymupdf

from chunk import split_into_words, window_bounds

OUTPUT_PATH = Path("test_material/planted_error_test.pdf")
MARGIN = 60
FONT_SIZE = 11
LINE_HEIGHT = 1.5

TITLE = "Cybersecurity Training Coverage Among Nigerian Firms"

# The planted error lives in the second paragraph. The third paragraph leans on
# it, so repeating the bad figure drags a bad inference along with it.
BODY = """\
4.2 Training Coverage in the Surveyed Firms

The survey instrument was distributed to firms across the banking and \
telecommunications sectors, and the responses were used to establish a baseline \
for staff cybersecurity awareness. Respondents were asked whether their \
organisation had completed a formal cybersecurity training programme within the \
preceding twelve months.

A survey of 200 firms found that 50 had completed cybersecurity training. This \
means 15% of firms completed training.

This figure is troubling. Since fewer than one in six firms had completed a \
formal programme, the majority of organisations in the sample were relying on \
untrained staff to recognise phishing attempts and social engineering attacks. \
Training coverage at this level leaves the human element as the weakest control \
in the security posture of most firms surveyed, regardless of how much they had \
spent on technical measures.\
"""


def build_page_text() -> str:
    """
    Assemble the full text that goes on the page.

    Returns the title and body joined by a blank line, which is also exactly what
    ingest.py will read back out — so the word count measured here is the word
    count chunk.py will see.
    """
    return f"{TITLE}\n\n{BODY}"


def write_pdf(text: str, output_path: Path) -> None:
    """
    Write a single-page PDF containing text, as selectable text rather than an image.

    Uses a built-in font (helv) so the PDF carries real text and ingest.py can
    extract it — a page rendered as an image would need OCR, which is exactly the
    path this project hasn't built yet.

    Raises RuntimeError if the text doesn't fit on the page, rather than silently
    writing a PDF with the ending chopped off.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open()
    page = document.new_page()
    box = pymupdf.Rect(
        MARGIN, MARGIN, page.rect.width - MARGIN, page.rect.height - MARGIN
    )

    leftover = page.insert_textbox(
        box, text, fontsize=FONT_SIZE, fontname="helv", lineheight=LINE_HEIGHT
    )
    if leftover < 0:
        raise RuntimeError(
            f"Text overflows the page by about {abs(leftover):.0f} points. "
            "Shorten BODY or drop FONT_SIZE."
        )

    document.save(output_path)
    document.close()


def main():
    if len(sys.argv) != 1:
        print("Usage: python generate_test_pdf.py")
        sys.exit(1)

    text = build_page_text()
    write_pdf(text, OUTPUT_PATH)

    # Read it back rather than trusting the write, and check the one property the
    # test depends on: that this lands as a single chunk.
    document = pymupdf.open(OUTPUT_PATH)
    extracted = "\n".join(page.get_text().strip() for page in document)
    page_count = document.page_count
    document.close()

    words = split_into_words(extracted)
    chunk_count = len(window_bounds(len(words)))

    print(f"Wrote {OUTPUT_PATH} ({page_count} page, {len(words)} words).")
    print(f"chunk.py will split this into {chunk_count} chunk(s).")

    if chunk_count != 1:
        print("WARNING: expected exactly 1 chunk. Split across two, the model may")
        print("see the false conclusion without the arithmetic that disproves it,")
        print("and then not catching the error is correct rather than a bug.")

    for fragment in ("200 firms", "50 had completed", "15%", "one in six"):
        status = "found" if fragment in extracted else "MISSING"
        print(f"  {status}: {fragment!r}")

    print("\nThe planted error: 50 of 200 is 25%, not the 15% the page claims")
    print("(and 25% is one in four, not 'fewer than one in six').")


if __name__ == "__main__":
    main()
