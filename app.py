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
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from pdf_to_audio import (
    PARAGRAPH_SILENCE_SEC,
    SAMPLE_RATE,
    extract_text,
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
async def upload_pdf(file: UploadFile):
    file_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{file_id}.pdf"
    save_path.write_bytes(await file.read())
    return {"id": file_id, "filename": file.filename}


@app.get("/uploads/{file_id}")
async def serve_upload(file_id: str):
    path = UPLOAD_DIR / f"{file_id}.pdf"
    if path.exists():
        return FileResponse(str(path), media_type="application/pdf")
    return JSONResponse({"error": "not found"}, status_code=404)


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
        if mode == "pdf":
            pdf_id = data.get("pdf_id")
            pdf_path = str(UPLOAD_DIR / f"{pdf_id}.pdf")
            await ws.send_json({"type": "status", "message": "Extracting text from PDF..."})
            raw_text = await asyncio.to_thread(extract_text, pdf_path)
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
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    print(f"Temp dir: {WORK_DIR}")
    print("Starting AudioPlayback at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
