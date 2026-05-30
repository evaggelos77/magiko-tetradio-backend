"""
Magikó Tetrádio AI Junior — backend.

Single small endpoint: POST /api/tts → mp3 audio from OpenAI's text-to-speech,
so the kids' app can speak Greek with a clean, premium female voice on every
device (phones, tablets, Windows desktops) instead of relying on whatever
Web Speech voice happens to be installed.
"""

import io
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# --- App + CORS ---------------------------------------------------------------
app = FastAPI(title="Magikó Tetrádio backend", version="1.0.0")

# Frontends that may call this backend.
ALLOWED_ORIGINS = [
    "https://magiko-tetradio.onrender.com",
    "https://evlabsai.gr",
    "https://www.evlabsai.gr",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- OpenAI client ------------------------------------------------------------
_client: Optional[OpenAI] = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise HTTPException(status_code=503, detail="TTS not configured")
        _client = OpenAI(api_key=api_key)
    return _client


# --- Models -------------------------------------------------------------------
ALLOWED_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
# "shimmer" is warm and feminine — a good default for a kids' AI in Greek.
DEFAULT_VOICE = "shimmer"
# Hard cap per request to keep cost predictable.
MAX_CHARS = 800


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


# --- Routes -------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "magiko-tetradio-backend",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    }


@app.post("/api/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    voice = (req.voice or DEFAULT_VOICE).strip().lower()
    if voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    try:
        client = _openai()
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
        )
        audio_bytes = response.read()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Don't leak the upstream error verbatim, but keep a hint useful in logs.
        print(f"[tts] upstream error: {exc!r}")
        raise HTTPException(status_code=502, detail="TTS upstream error")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )
