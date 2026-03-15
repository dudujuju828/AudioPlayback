#!/usr/bin/env python3
"""Convert PDF documents to audiobooks using Kokoro TTS."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 24000
PARAGRAPH_SILENCE_SEC = 0.5


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF, handling multi-column layouts via block sorting."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        texts = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
        pages.append("\n\n".join(texts))
    doc.close()
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Code-block detection helpers
# ---------------------------------------------------------------------------

def _is_code_line(line: str) -> bool:
    """Heuristically decide whether a line looks like source code or shell output."""
    s = line.strip()
    if not s:
        return False

    # Numbered code lines (e.g. "1  #include <stdio.h>")
    if re.match(r"^\d{1,3}\s{2,}\S", s):
        return True
    # Shell / REPL prompts
    if re.match(r"^[\$%>]\s", s) or re.match(r"^prompt>", s, re.IGNORECASE):
        return True
    # C preprocessor
    if re.match(r"^#\s*(include|define|ifdef|ifndef|endif|pragma)\b", s):
        return True
    # C declarations / keywords at start of line
    if re.match(
        r"^(int|void|char|float|double|unsigned|long|short|"
        r"struct|enum|typedef|static|extern|const)\s+\w", s
    ):
        return True
    # Lines ending with ; { } that are short (likely code, not prose)
    if re.search(r"[;{}]\s*$", s) and len(s) < 120:
        if not re.match(r"^[A-Z][a-z]", s):  # skip normal sentences
            return True
    # Common C library / system calls
    if re.match(
        r"^(printf|fprintf|sprintf|assert|exit|malloc|calloc|free|"
        r"fork|exec\w*|wait\w*|open|close|read|write|ioctl|mmap|"
        r"pthread_\w+|Pthread_\w+|fopen|fclose|fread|fwrite)\s*\(", s
    ):
        return True
    # x86 assembly mnemonics
    if re.match(
        r"^(mov|push|pop|call|ret|jmp|je|jne|jz|jnz|add|sub|"
        r"cmp|xor|lea|test|nop|int)\s", s, re.IGNORECASE
    ):
        return True
    return False


def _remove_code_blocks(text: str) -> str:
    """Strip runs of code / terminal output, keeping surrounding prose."""
    lines = text.split("\n")
    n = len(lines)
    is_code = [_is_code_line(l) for l in lines]

    # Blank lines sandwiched between code lines belong to the code block.
    for i in range(1, n - 1):
        if lines[i].strip() == "" and is_code[i - 1]:
            for j in range(i + 1, min(i + 4, n)):
                if lines[j].strip():
                    if is_code[j]:
                        is_code[i] = True
                    break

    # Bridge single non-code lines sitting between two code regions.
    for i in range(1, n - 1):
        if not is_code[i] and is_code[i - 1]:
            for j in range(i + 1, min(i + 3, n)):
                if is_code[j]:
                    is_code[i] = True
                    break

    return "\n".join(l for i, l in enumerate(lines) if not is_code[i])


# ---------------------------------------------------------------------------
# Text sanitisation
# ---------------------------------------------------------------------------

def sanitize_text(
    text: str, keep_headings: bool = True, keep_code: bool = False
) -> str:
    """Clean raw PDF text so it sounds natural when spoken."""

    # Fix hyphenated line breaks (e.g. "com-\nputer" -> "computer")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # ---- Headers / footers / page furniture ----
    for pat in [
        r"(?m)^.*OPERATING SYSTEMS.*\[VERSION.*\].*$",
        r"(?m)^.*THREE EASY PIECES.*$",
        r"(?m)^.*WWW\.OSTEP\.ORG.*$",
        r"(?m)^.*OSTEP\.ORG.*$",
        r"(?m)^\s*\u00a9.*$",          # © lines
        r"(?m)^\s*©.*$",
    ]:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Standalone page numbers
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # Figure / table captions
    text = re.sub(
        r"(?m)^(?:Figure|Table)\s+\d+[\.\:].+$", "", text, flags=re.IGNORECASE
    )

    # ---- Code blocks ----
    if not keep_code:
        text = _remove_code_blocks(text)

    # ---- Citations & footnotes ----
    # Academic tags: [PP03], [BOH+10], [K+61,L78], …
    text = re.sub(
        r"\[(?:[A-Z][A-Za-z+]*\d{2,4}"
        r"(?:\s*,\s*[A-Z][A-Za-z+]*\d{2,4})*)\]",
        "", text,
    )
    # Bracketed footnote numbers: [1], [12]
    text = re.sub(r"\[\d{1,3}\]", "", text)

    # ---- Back-matter sections (References, Homework) ----
    text = re.sub(
        r"(?s)\n\s*References\s*\n.+$", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"(?s)\n\s*Homework\s*(?:\(.*?\))?\s*\n.+$", "", text, flags=re.IGNORECASE
    )

    # ---- ASIDE / CRUX boxes — keep prose, drop label ----
    text = re.sub(
        r"(?m)^(?:ASIDE|TIP|THE CRUX OF THE PROBLEM|CRUX OF THE PROBLEM)"
        r"[:\s]*\n?",
        "", text,
    )

    # ---- Section headings ----
    if not keep_headings:
        text = re.sub(r"(?m)^\d+\.\d+[\.\d]*\s+.+$", "", text)

    # ---- Conservative abbreviation expansion ----
    for pat, repl in [
        (r"\be\.g\.\s",   "for example, "),
        (r"\bi\.e\.\s",   "that is, "),
        (r"\betc\.\b",    "etcetera"),
        (r"\bvs\.\s",     "versus "),
        (r"\bw\.r\.t\.\s","with respect to "),
    ]:
        text = re.sub(pat, repl, text)

    # ---- Whitespace normalisation ----
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    # ---- Markdown artefacts ----
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines, dropping empties."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

def generate_audiobook(
    text: str,
    output_path: str,
    voice: str = "af_bella",
    speed: float = 1.0,
    output_format: str = "mp3",
) -> None:
    from kokoro import KPipeline

    lang_code = voice[0] if voice[0] in "abjz" else "a"
    pipeline = KPipeline(lang_code=lang_code)

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        print("No text to convert.")
        return

    silence = np.zeros(int(PARAGRAPH_SILENCE_SEC * SAMPLE_RATE), dtype=np.float32)
    all_audio: list[np.ndarray] = []

    print(f"Generating audio for {len(paragraphs)} paragraphs...")
    for paragraph in tqdm(paragraphs, desc="Processing"):
        chunks: list[np.ndarray] = []
        try:
            for _gs, _ps, audio in pipeline(paragraph, voice=voice, speed=speed):
                if audio is not None:
                    chunks.append(audio)
        except Exception as e:
            tqdm.write(f"Warning: skipped paragraph — {e}")
            continue

        if chunks:
            all_audio.append(np.concatenate(chunks))
            all_audio.append(silence)

    if not all_audio:
        print("No audio was generated.")
        return

    # Drop trailing silence
    if len(all_audio) > 1:
        all_audio = all_audio[:-1]

    final_audio = np.concatenate(all_audio)
    out = Path(output_path)

    if output_format == "wav":
        sf.write(str(out), final_audio, SAMPLE_RATE)
    else:
        wav_tmp = out.with_suffix(".tmp.wav")
        sf.write(str(wav_tmp), final_audio, SAMPLE_RATE)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(wav_tmp),
                    "-codec:a", "libmp3lame", "-qscale:a", "2",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            wav_tmp.unlink()
        except FileNotFoundError:
            final_wav = out.with_suffix(".wav")
            wav_tmp.rename(final_wav)
            print(f"\nffmpeg not found — saved as WAV: {final_wav}")
            out = final_wav
        except subprocess.CalledProcessError as e:
            print(f"\nffmpeg error: {e.stderr.decode()}")
            print(f"WAV kept at: {wav_tmp}")
            return

    dur = len(final_audio) / SAMPLE_RATE
    print(f"\nDone! {out} ({int(dur // 60)}m {int(dur % 60)}s)")


# ---------------------------------------------------------------------------
# Quick installation test
# ---------------------------------------------------------------------------

def test_tts() -> bool:
    print("Running TTS test...")
    try:
        from kokoro import KPipeline

        pipe = KPipeline(lang_code="a")
        chunks = []
        for _gs, _ps, audio in pipe(
            "Hello, this is a test of the Kokoro text to speech system.",
            voice="af_bella",
        ):
            if audio is not None:
                chunks.append(audio)

        if chunks:
            dur = sum(len(c) for c in chunks) / SAMPLE_RATE
            print(f"Test passed! Generated {dur:.1f}s of audio.")
            return True

        print("Test failed: no audio produced.")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PDF document to an audiobook using Kokoro TTS",
    )
    parser.add_argument("input", nargs="?", help="Path to input PDF file")
    parser.add_argument("-o", "--output", help="Output audio file path")
    parser.add_argument(
        "--voice", default="af_bella", help="Kokoro voice (default: af_bella)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed (default: 1.0)"
    )
    parser.add_argument(
        "--keep-headings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Speak section headings (default: on)",
    )
    parser.add_argument(
        "--keep-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Speak code blocks (default: off)",
    )
    parser.add_argument(
        "--output-format",
        choices=["wav", "mp3"],
        default="mp3",
        help="Output format (default: mp3)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run a quick TTS test and exit"
    )

    args = parser.parse_args()

    if args.test:
        sys.exit(0 if test_tts() else 1)

    if not args.input:
        parser.error("the following arguments are required: input")

    pdf = Path(args.input)
    if not pdf.exists():
        print(f"Error: {pdf} not found")
        sys.exit(1)

    out = args.output or str(pdf.with_suffix(f".{args.output_format}"))

    # Step 1 — extract
    print(f"Extracting text from {pdf}...")
    raw = extract_text(str(pdf))
    print(f"Extracted {len(raw):,} characters")

    # Step 2 — sanitise
    print("Sanitizing text...")
    clean = sanitize_text(
        raw, keep_headings=args.keep_headings, keep_code=args.keep_code
    )
    print(f"Cleaned: {len(clean):,} characters")

    # Step 3 — generate audio
    generate_audiobook(
        clean,
        out,
        voice=args.voice,
        speed=args.speed,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
