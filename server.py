"""
Magikó Tetrádio AI Junior — backend.

Two small endpoints for the kids' app:

  POST /api/tts                → mp3 audio from OpenAI TTS so the AI can speak
                                 Greek with a clean, premium female voice on
                                 every device.

  POST /api/generate-question  → a single fresh educational question in Greek
                                 (multiple choice). Solves the "questions
                                 repeat after a small cycle" problem in the
                                 visual/word modules (shapes, colors, animals,
                                 letters, spelling, etc.). Supports a free-form
                                 parent topic so a parent can type e.g.
                                 "προπαίδεια του 7" and get tailored questions.
"""

import io
import json
import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# --- App + CORS ---------------------------------------------------------------
app = FastAPI(title="Magikó Tetrádio backend", version="1.1.0")

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
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        _client = OpenAI(api_key=api_key)
    return _client


# --- /api/health --------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "magiko-tetradio-backend",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    }


# --- /api/tts -----------------------------------------------------------------
ALLOWED_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
DEFAULT_VOICE = "shimmer"  # warm, feminine — friendly default for a kids' AI
MAX_TTS_CHARS = 800


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None  # 0.25 – 4.0, default 0.9 (gentler for kids)


@app.post("/api/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS]

    voice = (req.voice or DEFAULT_VOICE).strip().lower()
    if voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    # Slower default than OpenAI's 1.0 — clearer for a child's ear.
    speed = req.speed if (req.speed is not None) else 0.9
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 0.9
    speed = max(0.5, min(1.5, speed))

    try:
        client = _openai()
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
            speed=speed,
        )
        audio_bytes = response.read()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[tts] upstream error: {exc!r}")
        raise HTTPException(status_code=502, detail="TTS upstream error")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


# --- /api/generate-question ---------------------------------------------------
MAX_TOPIC_CHARS = 120
MAX_AVOID_ITEMS = 8
MAX_AVOID_CHAR_PER_ITEM = 200

# Friendly Greek labels per module so the prompt is grounded.
MODULE_HINTS = {
    "shapes":       "γεωμετρικά σχήματα (κύκλος, τρίγωνο, τετράγωνο, ορθογώνιο, γραμμή). Δώσε καθημερινό αντικείμενο που μοιάζει με σχήμα.",
    "colors":       "χρώματα. Δώσε ένα γνωστό αντικείμενο και ρώτα τι χρώμα έχει.",
    "animals":      "ζωάκια και οι ήχοι ή τα χαρακτηριστικά τους.",
    "letters":      "γράμματα του ελληνικού αλφαβήτου — π.χ. ποια λέξη αρχίζει από Λ.",
    "syllables":    "συλλαβές και αρχές λέξεων.",
    "numbers_kg":   "μέτρημα μικρών αριθμών (1-10) με εμφανή ποσότητα.",
    "addition":     "απλή πρόσθεση Νηπιαγωγείου/Α' Δημοτικού.",
    "subtraction":  "απλή αφαίρεση Νηπιαγωγείου/Α' Δημοτικού.",
    "multiplication":"πολλαπλασιασμός (προπαίδεια έως 9x9).",
    "fractions":    "απλά κλάσματα (1/2, 1/4, 3/4) με οπτικές αναπαραστάσεις πίτσας/μήλου.",
    "decimals":     "απλοί δεκαδικοί αριθμοί και αντιστοιχία με κλάσματα.",
    "problem":      "μικρό λεκτικό πρόβλημα της καθημερινότητας με αριθμητική.",
    "spelling":     "ορθογραφία απλών ελληνικών λέξεων — διάλεξε τη σωστή.",
    "spelling_easy":"ορθογραφία απλών ελληνικών λέξεων για μικρές τάξεις.",
    "dictation":    "υπαγόρευση: παρουσίασε λέξη και βρες τη σωστή ορθογραφία.",
    "truefalse":    "Σωστό ή Λάθος για απλή πρόταση γενικών γνώσεων ή μαθηματικών.",
    "truefalse_kg": "Σωστό ή Λάθος πολύ απλά, για μικρά παιδιά.",
    "grammar":      "γραμματική: αναγνώριση ρήματος/ουσιαστικού/επιθέτου.",
    "grammar_basic":"βασική γραμματική: ρήμα/ουσιαστικό/επίθετο για μικρές τάξεις.",
    "syntax":       "συντακτικό: βρες υποκείμενο/αντικείμενο/ρήμα.",
    "reading":      "ανάγνωση πρότασης και απλή κατανόηση.",
    "comprehension_adv": "κατανόηση κειμένου για μεγαλύτερες τάξεις.",
    "emotions":     "συναισθήματα — αναγνώριση και πρακτική αντιμετώπιση.",
    "prewriting":   "προγραφικές ασκήσεις παρατήρησης.",
    "listening_kg": "ακουστική αντίληψη για μικρά παιδιά.",
}

GRADE_HINTS = {
    "kg":  "Νηπιαγωγείο (4-5 ετών) — πολύ απλό, φιλικό λεξιλόγιο, εικονιστικό.",
    "a":   "Α' Δημοτικού (6-7 ετών).",
    "b":   "Β' Δημοτικού (7-8 ετών).",
    "c":   "Γ' Δημοτικού (8-9 ετών).",
    "d":   "Δ' Δημοτικού (9-10 ετών).",
    "e":   "Ε' Δημοτικού (10-11 ετών).",
    "f":   "ΣΤ' Δημοτικού (11-12 ετών).",
}

LEVEL_HINTS = {
    "easy":   "Πιο εύκολο και υποστηρικτικό.",
    "normal": "Ισορροπημένη δυσκολία.",
    "hard":   "Λίγο πιο απαιτητικό, αλλά πάντα φιλικό για το παιδί.",
}


class GenQuestionRequest(BaseModel):
    module: Optional[str] = None      # e.g. "shapes", "addition" — internal id
    moduleTitle: Optional[str] = None # Greek title shown to the kid (optional)
    grade: Optional[str] = None       # "kg" | "a"..."f"
    level: Optional[str] = "normal"   # "easy"|"normal"|"hard"
    topic: Optional[str] = None       # parent's free-text topic (optional)
    avoid: Optional[list[str]] = None # previous question texts to not repeat


def _sanitize_avoid(items: Optional[list[str]]) -> list[str]:
    if not items:
        return []
    cleaned: list[str] = []
    for it in items[:MAX_AVOID_ITEMS]:
        if not isinstance(it, str):
            continue
        s = it.strip()
        if not s:
            continue
        if len(s) > MAX_AVOID_CHAR_PER_ITEM:
            s = s[:MAX_AVOID_CHAR_PER_ITEM]
        cleaned.append(s)
    return cleaned


SYSTEM_PROMPT = """\
Είσαι σχεδιάστρια εκπαιδευτικού περιεχομένου για ελληνική παιδική εφαρμογή.
Δημιουργείς ΜΙΑ ερώτηση πολλαπλής επιλογής, στα Ελληνικά, ασφαλή και χαρούμενη.

Κανόνες:
- Όλα τα κείμενα στα ΕΛΛΗΝΙΚΑ.
- Ακριβώς 4 επιλογές απάντησης, μία σωστή.
- Καμία επιλογή δεν επαναλαμβάνεται.
- Πάντα κατάλληλο για παιδί της δηλωμένης τάξης/ηλικίας.
- Σύντομη, καθαρή διατύπωση. Χωρίς ειρωνεία ή φόβο.
- ΠΟΤΕ να μην επαναλάβεις μία από τις ερωτήσεις που υπάρχουν στο πεδίο "avoid".
- Απάντησε ΜΟΝΟ με έγκυρο JSON, ακριβώς της παρακάτω δομής, χωρίς prefix/explanation."""

JSON_SHAPE = """\
{
  "question": "Η ερώτηση στα Ελληνικά (μία πρόταση).",
  "emoji": "1-3 emojis που εικονογραφούν (μπορεί να είναι κενό)",
  "answers": ["επιλογή1", "επιλογή2", "επιλογή3", "επιλογή4"],
  "correct": 0,
  "help": "Σύντομη μία πρόταση εξήγηση γιατί η σωστή είναι σωστή.",
  "tag": "μία λέξη/φράση κατηγορίας, π.χ. 'σχήματα'"
}"""


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from the model's reply."""
    text = (text or "").strip()
    # strip fenced blocks like ```json ... ```
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except Exception:
        # try to find the first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def _validate_question(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    q = (obj.get("question") or "").strip()
    answers = obj.get("answers") or []
    correct = obj.get("correct")
    if not q:
        raise ValueError("missing question")
    if not isinstance(answers, list) or len(answers) != 4:
        raise ValueError("answers must be exactly 4")
    cleaned_answers = []
    seen = set()
    for a in answers:
        s = str(a).strip()
        if not s or s in seen:
            raise ValueError("answers must be unique and non-empty")
        seen.add(s)
        cleaned_answers.append(s)
    if not isinstance(correct, int) or correct < 0 or correct > 3:
        raise ValueError("correct must be integer 0..3")
    emoji = str(obj.get("emoji") or "").strip()
    help_txt = str(obj.get("help") or "").strip()
    tag = (str(obj.get("tag") or "").strip() or "γενική")[:40]
    return {
        "question": q,
        "emoji": emoji,
        "answers": cleaned_answers,
        "correct": correct,
        "help": help_txt,
        "tag": tag,
    }


@app.post("/api/generate-question")
def generate_question(req: GenQuestionRequest):
    module = (req.module or "").strip().lower()
    module_hint = MODULE_HINTS.get(module, "")
    title = (req.moduleTitle or "").strip()
    grade_hint = GRADE_HINTS.get((req.grade or "").strip().lower(), "")
    level_hint = LEVEL_HINTS.get((req.level or "normal").strip().lower(), LEVEL_HINTS["normal"])

    topic = (req.topic or "").strip()
    if len(topic) > MAX_TOPIC_CHARS:
        topic = topic[:MAX_TOPIC_CHARS]

    avoid = _sanitize_avoid(req.avoid)
    avoid_block = ""
    if avoid:
        avoid_block = "\nΕρωτήσεις που ΔΕΝ πρέπει να επαναληφθούν (avoid):\n" + "\n".join(f"- {a}" for a in avoid)

    parts = []
    if title:   parts.append(f"Θεματική ενότητα: {title}.")
    if module_hint:
        parts.append(f"Λεπτομέρειες θεματικής: {module_hint}")
    if grade_hint:
        parts.append(f"Επίπεδο τάξης: {grade_hint}")
    parts.append(f"Δυσκολία: {level_hint}")
    if topic:
        parts.append(f"Ειδικό αίτημα γονέα: «{topic}». Αν ταιριάζει στη θεματική, δώσε προτεραιότητα.")

    user_prompt = (
        "Δημιούργησε ΜΙΑ νέα ερώτηση πολλαπλής επιλογής για το παιδί.\n\n"
        + "\n".join(parts)
        + "\n\nΔομή JSON που πρέπει να επιστρέψεις (μόνο JSON, χωρίς σχόλια):\n"
        + JSON_SHAPE
        + avoid_block
    )

    try:
        client = _openai()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        raw = completion.choices[0].message.content or ""
        data = _extract_json(raw)
        return _validate_question(data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[generate-question] error: {exc!r}")
        # Surface the upstream message so we can see what's wrong in the UI
        raise HTTPException(status_code=502, detail=f"OpenAI: {type(exc).__name__}: {str(exc)[:280]}")
