#!/usr/bin/env python3
"""Convert documents to audiobooks using Kokoro TTS.

Supports PDF, EPUB, DOCX, HTML, and plain text input.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 24000
PARAGRAPH_SILENCE_SEC = 0.5

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".docx", ".html", ".htm", ".txt", ".md"}


# ---------------------------------------------------------------------------
# Extraction — PDF
# ---------------------------------------------------------------------------

def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF.

    Handles multi-column layouts via block sorting and automatically strips
    repeated headers/footers by detecting text that appears in the margin
    zones (top/bottom 12%) of many pages.
    """
    import fitz  # PyMuPDF
    from collections import Counter

    doc = fitz.open(pdf_path)

    # First pass — collect blocks with position metadata.
    page_blocks: list[list[dict]] = []
    for page in doc:
        height = page.rect.height
        blocks = page.get_text("blocks", sort=True)
        page_blocks.append([
            {"text": b[4].strip(), "y0": b[1], "y1": b[3], "height": height}
            for b in blocks if b[6] == 0 and b[4].strip()
        ])

    # Second pass — detect repeated margin text (headers / footers).
    repeated: set[str] = set()
    if len(page_blocks) >= 4:
        margin_counts: Counter[str] = Counter()
        for blocks in page_blocks:
            seen: set[str] = set()
            for b in blocks:
                if b["y0"] / b["height"] < 0.12 or b["y1"] / b["height"] > 0.88:
                    t = b["text"]
                    if t not in seen:
                        margin_counts[t] += 1
                        seen.add(t)
        threshold = len(page_blocks) * 0.4
        repeated = {t for t, c in margin_counts.items() if c >= threshold}

    # Third pass — build final text, excluding repeated headers/footers.
    pages: list[str] = []
    for blocks in page_blocks:
        texts = [b["text"] for b in blocks if b["text"] not in repeated]
        pages.append("\n\n".join(texts))

    doc.close()
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Extraction — EPUB, DOCX, HTML, plain text
# ---------------------------------------------------------------------------

def extract_text_epub(epub_path: str) -> str:
    """Extract text from an EPUB ebook."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "EPUB support requires extra packages: "
            "pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(epub_path, options={"ignore_ncx": True})
    chapters: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            chapters.append(text.strip())
    return "\n\n".join(chapters)


def extract_text_docx(docx_path: str) -> str:
    """Extract text from a Word document."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "DOCX support requires python-docx: pip install python-docx"
        )

    doc = Document(docx_path)
    paragraphs: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def extract_text_html(html_path: str) -> str:
    """Extract readable text from an HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "HTML support requires beautifulsoup4: pip install beautifulsoup4"
        )

    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def extract_text_plaintext(file_path: str) -> str:
    """Read a plain text or Markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text_from_file(file_path: str) -> str:
    """Extract text from any supported document format."""
    ext = Path(file_path).suffix.lower()
    extractors = {
        ".pdf": extract_text,
        ".epub": extract_text_epub,
        ".docx": extract_text_docx,
        ".html": extract_text_html,
        ".htm": extract_text_html,
        ".txt": extract_text_plaintext,
        ".md": extract_text_plaintext,
    }
    extractor = extractors.get(ext)
    if extractor is None:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return extractor(file_path)


# ---------------------------------------------------------------------------
# Code-block detection helpers
# ---------------------------------------------------------------------------

def _is_code_line(line: str) -> bool:
    """Heuristically decide whether a line looks like source code or shell output.

    Covers C/C++, Python, Java, JavaScript, shell, Go, Rust, and assembly.
    """
    s = line.strip()
    if not s:
        return False

    # Numbered code lines (e.g. "1  #include <stdio.h>")
    if re.match(r"^\d{1,3}\s{2,}\S", s):
        return True
    # Shell / REPL prompts
    if re.match(r"^[\$%>]{1,3}\s", s) or re.match(r"^(prompt|>>>|\.\.\.)\s*", s, re.IGNORECASE):
        return True
    # C / C++ preprocessor
    if re.match(r"^#\s*(include|define|ifdef|ifndef|endif|pragma|import)\b", s):
        return True
    # C / C++ / Java / Go type declarations at start of line
    if re.match(
        r"^(int|void|char|float|double|unsigned|long|short|bool|auto|"
        r"struct|enum|typedef|static|extern|const|class|public|private|"
        r"protected|return|package|import|func|fn|let|mut|pub|use)\s+\w", s
    ):
        return True
    # Python-style def / class / import / from-import
    if re.match(r"^(def|class|import|from)\s+\w", s):
        return True
    # Lines ending with ; { } that are short (likely code, not prose)
    if re.search(r"[;{}]\s*$", s) and len(s) < 120:
        if not re.match(r"^[A-Z][a-z]", s):  # skip normal sentences
            return True
    # Function call at start of line (common across C, Python, JS, etc.)
    if re.match(
        r"^(printf|fprintf|sprintf|assert|exit|malloc|calloc|free|"
        r"fork|exec\w*|wait\w*|open|close|read|write|ioctl|mmap|"
        r"pthread_\w+|Pthread_\w+|fopen|fclose|fread|fwrite|"
        r"print|console\.log|fmt\.Print|System\.out)\s*\(", s
    ):
        return True
    # x86 / ARM assembly mnemonics
    if re.match(
        r"^(mov|push|pop|call|ret|jmp|je|jne|jz|jnz|add|sub|"
        r"cmp|xor|lea|test|nop|int|ldr|str|bl|bx)\s", s, re.IGNORECASE
    ):
        return True
    # Lines that are mostly non-alphabetic (operators, brackets, etc.)
    alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
    if len(s) > 4 and alpha_ratio < 0.3:
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
    """Clean extracted document text so it sounds natural when spoken.

    Works across PDFs, ebooks, web pages, and other formats — no
    assumptions about any specific document or publisher.
    """

    # Fix hyphenated line breaks (e.g. "com-\nputer" → "computer")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # ---- Page furniture ----
    # Copyright lines
    text = re.sub(r"(?m)^\s*[\u00a9©].*$", "", text)
    # Standalone page numbers
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # "Page X of Y" patterns
    text = re.sub(r"(?m)^.*Page\s+\d+\s+of\s+\d+.*$", "", text, flags=re.IGNORECASE)
    # All-caps lines that are short (usually running headers)
    text = re.sub(r"(?m)^[A-Z\s\d\-:,]{8,80}$", _drop_if_allcaps, text)

    # ---- Figure / table / exhibit captions ----
    text = re.sub(
        r"(?m)^(?:Figure|Fig\.|Table|Chart|Diagram|Exhibit)\s+\d+[\.\:\-].+$",
        "", text, flags=re.IGNORECASE,
    )

    # ---- URLs and email addresses ----
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # ---- Table of contents lines ----
    # "Some Title ........... 42" or "Some Title          42"
    text = re.sub(r"(?m)^.{3,60}\.{4,}\s*\d+\s*$", "", text)
    text = re.sub(r"(?m)^.{3,60}\s{4,}\d{1,4}\s*$", "", text)

    # ---- Code blocks ----
    if not keep_code:
        text = _remove_code_blocks(text)

    # ---- Citations & footnotes ----
    # Academic citation tags: [PP03], [BOH+10], [K+61,L78]
    text = re.sub(
        r"\[(?:[A-Z][A-Za-z+]*\d{2,4}"
        r"(?:\s*,\s*[A-Z][A-Za-z+]*\d{2,4})*)\]",
        "", text,
    )
    # Bracketed footnote numbers: [1], [12]
    text = re.sub(r"\[\d{1,3}\]", "", text)
    # Superscript-style footnote markers ("1Of course" → "Of course")
    text = re.sub(r"(?m)^\d{1,2}(?=[A-Z][a-z])", "", text)

    # ---- Back-matter sections (cut from heading to end of text) ----
    for heading in [
        r"References", r"Bibliography", r"Works\s+Cited",
        r"Homework(?:\s*\(.*?\))?", r"Exercises",
    ]:
        text = re.sub(
            rf"(?s)\n\s*{heading}\s*\n.+$", "", text, flags=re.IGNORECASE,
        )

    # ---- Section headings ----
    if not keep_headings:
        # Numbered headings: "1.2 Title", "1.2.3 Title"
        text = re.sub(r"(?m)^\d+[\.\d]*\s+.{0,80}$", "", text)
        # "Chapter N" style
        text = re.sub(r"(?m)^Chapter\s+\d+.*$", "", text, flags=re.IGNORECASE)

    # ---- Abbreviation expansion ----
    for pat, repl in [
        (r"\be\.g\.\s",    "for example, "),
        (r"\bi\.e\.\s",    "that is, "),
        (r"\betc\.\b",     "etcetera"),
        (r"\bvs\.\s",      "versus "),
        (r"\bw\.r\.t\.\s", "with respect to "),
        (r"\bfig\.\s",     "figure "),
        (r"\beq\.\s",      "equation "),
    ]:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # ---- Common symbols → spoken equivalents ----
    _SYMBOL_MAP = {
        "\u2192": " to ",        # →
        "\u2190": " from ",      # ←
        "\u2248": " approximately equal to ",  # ≈
        "\u2264": " less than or equal to ",   # ≤
        "\u2265": " greater than or equal to ",# ≥
        "\u2260": " not equal to ",            # ≠
        "\u221e": " infinity ",  # ∞
        "\u00b0": " degrees ",   # °
        "\u00b1": " plus or minus ",           # ±
        "\u00d7": " times ",     # ×
        "\u00f7": " divided by ",# ÷
        "\u2014": ", ",          # em dash
        "\u2013": " to ",        # en dash (often used for ranges)
    }
    for sym, word in _SYMBOL_MAP.items():
        text = text.replace(sym, word)

    # ---- Bullet / list markers ----
    text = re.sub(
        r"(?m)^[\u2022\u2023\u25e6\u2043\u2219\u25cf\u25cb\u25aa\u25ab\u2726\u2727]\s*",
        "", text,
    )

    # ---- Whitespace normalisation ----
    text = re.sub(r"\n{3,}", "\n\n", text)       # 3+ newlines → paragraph break
    text = re.sub(r"[ \t]+", " ", text)           # collapse runs of spaces
    # Single newlines are just line-wraps — join them into prose.
    # Double newlines (paragraph breaks) are preserved.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # ---- Markdown artefacts ----
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)   # # Heading → Heading

    return text.strip()


def _drop_if_allcaps(m: re.Match) -> str:
    """Helper for all-caps header removal — only drop if actually all-caps."""
    line = m.group(0).strip()
    letters = [c for c in line if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return ""
    return m.group(0)


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
    fmts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    parser = argparse.ArgumentParser(
        description="Convert a document to an audiobook using Kokoro TTS",
    )
    parser.add_argument(
        "input", nargs="?",
        help=f"Path to input document ({fmts})",
    )
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

    infile = Path(args.input)
    if not infile.exists():
        print(f"Error: {infile} not found")
        sys.exit(1)
    if infile.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Error: unsupported format {infile.suffix}")
        print(f"Supported: {fmts}")
        sys.exit(1)

    out = args.output or str(infile.with_suffix(f".{args.output_format}"))

    # Step 1 — extract
    print(f"Extracting text from {infile}...")
    raw = extract_text_from_file(str(infile))
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
