"""FastAPI web interface for the PDF-to-audiobook pipeline."""

import asyncio
import queue as thread_queue
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from pdf_to_audio import (
    PARAGRAPH_SILENCE_SEC,
    SAMPLE_RATE,
    SUPPORTED_EXTENSIONS,
    extract_text_from_file,
    sanitize_text,
    split_paragraphs,
)

app = FastAPI()

WORK_DIR = Path(tempfile.mkdtemp(prefix="audioplayback_"))
UPLOAD_DIR = WORK_DIR / "uploads"
AUDIO_DIR = WORK_DIR / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

_voices_cache: list[str] | None = None


def get_available_voices() -> list[str]:
    global _voices_cache
    if _voices_cache is not None:
        return _voices_cache
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files("hexgrad/Kokoro-82M")
        _voices_cache = sorted(
            f.split("/")[-1].replace(".pt", "")
            for f in files if f.startswith("voices/") and f.endswith(".pt")
        )
    except Exception:
        _voices_cache = [
            "af_bella", "am_adam", "am_michael", "am_onyx", "am_puck",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        ]
    return _voices_cache


def generate_audiobook_web(
    text: str,
    output_path: str,
    voice: str = "am_michael",
    speed: float = 1.0,
    progress_callback=None,
) -> list[dict] | None:
    """Generate audiobook with timecode tracking and progress callbacks."""
    from kokoro import KPipeline

    lang_code = voice[0] if voice[0] in "abjz" else "a"
    pipeline = KPipeline(lang_code=lang_code)

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return None

    silence = np.zeros(int(PARAGRAPH_SILENCE_SEC * SAMPLE_RATE), dtype=np.float32)
    all_audio: list[np.ndarray] = []
    timecodes: list[dict] = []
    offset = 0.0
    total = len(paragraphs)

    for i, paragraph in enumerate(paragraphs):
        if progress_callback:
            progress_callback(f"Generating audio chunk {i + 1}/{total}...")

        chunks: list[np.ndarray] = []
        try:
            for _gs, _ps, audio in pipeline(paragraph, voice=voice, speed=speed):
                if audio is not None:
                    chunks.append(audio)
        except Exception:
            continue

        if chunks:
            para_audio = np.concatenate(chunks)
            duration = len(para_audio) / SAMPLE_RATE
            timecodes.append({
                "text": paragraph,
                "start": round(offset, 3),
                "end": round(offset + duration, 3),
            })
            all_audio.append(para_audio)
            all_audio.append(silence)
            offset += duration + PARAGRAPH_SILENCE_SEC

    if not all_audio:
        return None

    if len(all_audio) > 1:
        all_audio = all_audio[:-1]

    if progress_callback:
        progress_callback("Encoding audio file...")

    final_audio = np.concatenate(all_audio)
    out = Path(output_path)
    wav_tmp = out.with_suffix(".tmp.wav")
    sf.write(str(wav_tmp), final_audio, SAMPLE_RATE)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_tmp),
             "-codec:a", "libmp3lame", "-qscale:a", "2", str(out)],
            check=True, capture_output=True,
        )
        wav_tmp.unlink()
    except (FileNotFoundError, subprocess.CalledProcessError):
        wav_tmp.rename(out.with_suffix(".wav"))

    return timecodes


# ---- Routes ----

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/voices")
async def voices():
    return get_available_voices()


@app.post("/upload")
async def upload_file(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported file type: {ext}"},
            status_code=400,
        )
    file_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    save_path.write_bytes(await file.read())
    return {"id": file_id, "filename": file.filename, "ext": ext}


_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".epub": "application/epub+zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/plain",
}


@app.get("/uploads/{file_id}")
async def serve_upload(file_id: str):
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        return JSONResponse({"error": "not found"}, status_code=404)
    path = matches[0]
    media = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(path), media_type=media)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    path = AUDIO_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    media = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(str(path), media_type=media)


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_json()
        mode = data.get("mode", "text")
        voice = data.get("voice", "am_michael")
        speed = float(data.get("speed", 1.0))
        keep_headings = data.get("keep_headings", True)
        keep_code = data.get("keep_code", False)

        # Extract
        file_ext = None
        if mode in ("pdf", "file"):
            file_id = data.get("pdf_id")
            matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
            if not matches:
                await ws.send_json({"type": "error", "message": "Uploaded file not found"})
                return
            file_path = str(matches[0])
            file_ext = matches[0].suffix.lower()
            await ws.send_json({"type": "status", "message": "Extracting text..."})
            raw_text = await asyncio.to_thread(extract_text_from_file, file_path)
            await ws.send_json({"type": "status", "message": f"Extracted {len(raw_text):,} characters"})
        else:
            raw_text = data.get("text", "")
            if not raw_text.strip():
                await ws.send_json({"type": "error", "message": "No text provided"})
                return

        # Sanitize
        await ws.send_json({"type": "status", "message": "Sanitizing text..."})
        clean = await asyncio.to_thread(sanitize_text, raw_text, keep_headings, keep_code)
        await ws.send_json({"type": "status", "message": f"Cleaned text: {len(clean):,} characters"})

        if not clean.strip():
            await ws.send_json({"type": "error", "message": "No text remaining after cleanup"})
            return

        # Generate
        audio_id = uuid.uuid4().hex[:8]
        audio_path = str(AUDIO_DIR / f"{audio_id}.mp3")

        q: thread_queue.Queue[str] = thread_queue.Queue()

        gen_task = asyncio.create_task(
            asyncio.to_thread(
                generate_audiobook_web, clean, audio_path,
                voice, speed, lambda msg: q.put(msg),
            )
        )

        while not gen_task.done():
            await asyncio.sleep(0.15)
            while not q.empty():
                await ws.send_json({"type": "status", "message": q.get_nowait()})

        while not q.empty():
            await ws.send_json({"type": "status", "message": q.get_nowait()})

        try:
            timecodes = gen_task.result()
        except Exception as e:
            await ws.send_json({"type": "error", "message": str(e)})
            return

        if timecodes is None:
            await ws.send_json({"type": "error", "message": "Audio generation failed"})
            return

        # Find actual output file
        audio_name = None
        for ext in [".mp3", ".wav"]:
            if (AUDIO_DIR / f"{audio_id}{ext}").exists():
                audio_name = f"{audio_id}{ext}"
                break

        await ws.send_json({
            "type": "done",
            "audio_url": f"/audio/{audio_name}",
            "timecodes": timecodes,
            "clean_text": clean,
            "pdf_id": data.get("pdf_id"),
            "file_ext": file_ext if mode in ("pdf", "file") else None,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---- Quiz generation ----

QUIZ_PROMPT = """You are an expert educator creating a high-quality assessment from \
educational material. Your goal is to produce questions that genuinely deepen \
understanding — the kind a student would learn from even by getting wrong.

Create exactly 30 multiple-choice questions across three tiers:
- Questions 1-10: FOUNDATIONAL — test whether core concepts are understood correctly. \
Use scenarios that expose common misconceptions. Example: "A process calls fork(). \
Which of the following is true about the child process?"
- Questions 11-20: APPLIED — require combining two or more concepts to reason through \
a realistic scenario. Example: "A program maps a shared page, then one process writes \
to it. Under copy-on-write, what happens to physical memory?"
- Questions 21-30: ANALYTICAL — require deeper reasoning, tradeoff analysis, or \
predicting system behavior under unusual conditions. Example: "A scheduler uses \
round-robin with a 10ms quantum. Given these three CPU-bound jobs arriving at t=0, \
what is the average turnaround time?"

Question quality rules:
- Every wrong option must be a plausible misconception, not an obvious throwaway. A \
student who picks a wrong answer should have a specific misunderstanding you can address.
- The explanation must teach — state WHY the right answer is right AND what \
misconception each wrong answer represents if chosen.
- Never ask "which of these is a definition of X" — always require applying the concept.

CRITICAL formatting rules for answer options:
- All 4 options must be the same length and style — a reader must NOT be able to \
identify the correct answer by noticing one option is longer, more specific, or \
more carefully qualified than the others.
- If one option is a short phrase, ALL must be short phrases.
- If one option is a full sentence with technical detail, ALL must match that style.
- Avoid "all of the above" or "none of the above".
- Randomize which position (0-3) is correct — distribute roughly evenly.

Return ONLY a JSON array (no markdown, no code fences) where each element has:
- "question": the scenario or problem
- "options": array of exactly 4 answers, all visually uniform
- "correct": zero-based index of the correct option (0-3)
- "explanation": 2-3 sentences — why correct, and what misconception the best \
distractor targets

Text:
"""


class QuizRequest(BaseModel):
    text: str


def _run_quiz_generation(text: str) -> list[dict]:
    """Run claude CLI locally to generate quiz questions."""
    import json
    proc = subprocess.run(
        "claude -p",
        input=QUIZ_PROMPT + text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        shell=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "claude exited with an error")
    raw = proc.stdout.strip()
    # Extract JSON array from response (handles markdown fences, preamble, etc.)
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON array found in claude response")
    questions = json.loads(raw[start:end])

    # Shuffle option positions so correct answer is evenly distributed
    import random
    for q in questions:
        opts = q["options"]
        correct_text = opts[q["correct"]]
        random.shuffle(opts)
        q["correct"] = opts.index(correct_text)

    return questions


@app.post("/quiz/generate")
async def generate_quiz(req: QuizRequest):
    try:
        questions = await asyncio.to_thread(_run_quiz_generation, req.text)
        return questions
    except FileNotFoundError:
        return JSONResponse(
            {"error": "claude CLI not found — is Claude Code installed?"}, status_code=500,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Quiz generation timed out"}, status_code=500)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    print(f"Temp dir: {WORK_DIR}")
    print("Starting AudioPlayback at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
