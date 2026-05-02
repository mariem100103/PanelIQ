from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import whisper
import subprocess
import os
import sys
import shutil
import json
import time
import threading
from groq import Groq
from pathlib import Path
from typing import Optional
import imageio_ffmpeg
import numpy as np
import whisper.audio as _whisper_audio
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

# Whisper.audio.load_audio hard-codes the literal "ffmpeg" command which
# does not exist as such on Windows when using imageio_ffmpeg's bundled
# binary (named like "ffmpeg-win-x86_64-vN.N.exe"). Patch it so Whisper
# uses the binary we already rely on for our own ffmpeg calls.
def _whisper_load_audio(file: str, sr: int = _whisper_audio.SAMPLE_RATE):
    cmd = [
        FFMPEG_BIN,
        "-nostdin", "-threads", "0", "-i", file,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to load audio: {e.stderr.decode(errors='ignore')}"
        ) from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

_whisper_audio.load_audio = _whisper_load_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT       = Path(__file__).parent.parent
ENV_FILE   = ROOT / ".env"
VIDEO_FILE = ROOT / "videos" / "fsf_ep36.mp4"
AUDIO_FILE = ROOT / "videos" / "api_audio.wav"

if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("GROQ_API_KEY not set")

LIVE_STREAM_URL = os.environ.get("LIVE_STREAM_URL", "")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")

client = Groq(api_key=groq_key)
model = whisper.load_model(WHISPER_MODEL)

_hls_cache: dict[str, tuple[str, float]] = {}
_hls_cache_lock = threading.Lock()
HLS_CACHE_TTL_S = 3 * 3600

def _yt_dlp_argv() -> list[str]:
    """Prefer yt-dlp on PATH; otherwise same-interpreter module (Windows-friendly)."""
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if exe:
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]

def resolve_hls_url(youtube_url: str) -> str:
    now = time.time()
    with _hls_cache_lock:
        cached = _hls_cache.get(youtube_url)
        if cached and (now - cached[1]) < HLS_CACHE_TTL_S:
            return cached[0]
    cmd = _yt_dlp_argv() + [
        "--get-url", "-f",
        "best[protocol=m3u8_native]/best[protocol^=m3u8]/best",
        "--no-playlist", "--no-warnings",
        youtube_url,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")
    hls_url = result.stdout.strip().splitlines()[0]
    with _hls_cache_lock:
        _hls_cache[youtube_url] = (hls_url, time.time())
    return hls_url

def extract_audio_from_live(hls_url: str, out_path: Path) -> None:
    result = subprocess.run(
        [FFMPEG_BIN, "-i", hls_url, "-t", "30",
         "-vn", "-ar", "16000", "-ac", "1", str(out_path), "-y"],
        capture_output=True, timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg live failed: " + result.stderr.decode())

def extract_audio_from_file(video_path: Path, start_time: str, out_path: Path) -> None:
    result = subprocess.run(
        [FFMPEG_BIN, "-i", str(video_path),
         "-ss", start_time, "-t", "30",
         "-vn", "-ar", "16000", "-ac", "1", str(out_path), "-y"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg file failed: " + result.stderr.decode())

def _clean_llm_json(content: str) -> str:
    """Strip optional ```json ... ``` fences from model output."""
    s = content.strip()
    if not s.startswith("```"):
        return s
    s = s[3:].lstrip()
    if s.lower().startswith("json"):
        s = s[4:].lstrip()
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3].rstrip()
    return s

@app.get("/")
def root():
    return {"status": "QuizLive API running",
            "video_exists": VIDEO_FILE.exists(),
            "live_stream_url": LIVE_STREAM_URL or None,
            "cached_streams": len(_hls_cache)}

@app.get("/stream-status")
def stream_status(stream_url: Optional[str] = Query(default=None)):
    url_to_check = stream_url or LIVE_STREAM_URL
    if not url_to_check:
        return {"ok": False, "error": "No stream_url set", "video_exists": VIDEO_FILE.exists()}
    with _hls_cache_lock:
        cached = _hls_cache.get(url_to_check)
    if cached:
        age_s = int(time.time() - cached[1])
        return {"ok": True, "youtube_url": url_to_check,
                "hls_url": cached[0][:80]+"...", "cache_age_seconds": age_s}
    try:
        hls_url = resolve_hls_url(url_to_check)
        return {"ok": True, "youtube_url": url_to_check, "hls_url": hls_url[:80]+"..."}
    except Exception as e:
        return {"ok": False, "youtube_url": url_to_check, "error": str(e),
                "video_exists": VIDEO_FILE.exists()}

@app.get("/generate-question")
def generate_question(
    start_time: str = Query(default="00:00:30"),
    stream_url: Optional[str] = Query(default=None),
):
    effective_stream = stream_url or LIVE_STREAM_URL or None
    source_label: str
    hls_url = None

    def _transcribe_safely() -> str:
        """Run Whisper, surviving silent / music-only segments."""
        try:
            res = model.transcribe(str(AUDIO_FILE), language="ar")
        except RuntimeError:
            return ""
        return (res.get("text") or "").strip()

    if effective_stream:
        source_label = f"live:{effective_stream[:50]}"
        try:
            hls_url = resolve_hls_url(effective_stream)
        except Exception as e:
            if not VIDEO_FILE.exists():
                raise HTTPException(status_code=503, detail=f"yt-dlp failed: {e}")
            source_label = f"live_fallback:{VIDEO_FILE.name}"

        if hls_url:
            try:
                extract_audio_from_live(hls_url, AUDIO_FILE)
            except Exception:
                with _hls_cache_lock:
                    _hls_cache.pop(effective_stream, None)
                try:
                    hls_url = resolve_hls_url(effective_stream)
                    extract_audio_from_live(hls_url, AUDIO_FILE)
                except Exception as e2:
                    if not VIDEO_FILE.exists():
                        raise HTTPException(status_code=503, detail=str(e2))
                    hls_url = None
                    source_label = f"live_fallback:{VIDEO_FILE.name}"

        if hls_url is None:
            if not VIDEO_FILE.exists():
                raise HTTPException(status_code=500, detail="No video file found")
            extract_audio_from_file(VIDEO_FILE, start_time, AUDIO_FILE)

        transcript = _transcribe_safely()
        # Live audio can land on music / silence — fall back to local file then.
        if not transcript and VIDEO_FILE.exists():
            extract_audio_from_file(VIDEO_FILE, start_time, AUDIO_FILE)
            source_label = f"live_fallback:{VIDEO_FILE.name}"
            transcript = _transcribe_safely()
    else:
        source_label = f"file:{VIDEO_FILE.name}"
        if not VIDEO_FILE.exists():
            raise HTTPException(status_code=500, detail=f"Video not found: {VIDEO_FILE}")
        extract_audio_from_file(VIDEO_FILE, start_time, AUDIO_FILE)
        transcript = _transcribe_safely()

    if not transcript:
        raise HTTPException(status_code=500, detail="Whisper returned empty transcript")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "Tu es un générateur de questions de quiz pour une émission tunisienne.\n"
                "À partir d'une transcription audio (parfois bruitée), génère TOUJOURS UNE question QCM "
                "en arabe, même si la transcription est partielle — invente une question plausible "
                "sur la culture/musique/sport tunisien si nécessaire.\n"
                "Ne jamais refuser. Ne pas inclure de phrases d'excuse.\n"
                "Réponds STRICTEMENT en JSON valide avec ces clés:\n"
                "{\n"
                '  "guest_name": "nom ou null",\n'
                '  "question": "la question en arabe",\n'
                '  "options": ["option1", "option2", "option3", "option4"],\n'
                '  "correct_index": 0,\n'
                '  "category": "ترفيه/رياضة/سياسة",\n'
                '  "confidence": 0.85\n'
                "}")},
            {"role": "user", "content": f"Transcription: {transcript}"},
        ],
    )

    raw = _clean_llm_json(response.choices[0].message.content or "")
    try:
        output = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Groq returned non-JSON: {e}; snippet={raw[:400]!r}",
        ) from e
    output["transcript"] = transcript
    output["source"] = source_label
    return output