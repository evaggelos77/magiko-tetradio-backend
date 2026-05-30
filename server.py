"""
Magikó Tetrádio AI Junior — backend (TTS + AI questions + freemium plans).

Endpoints:
  GET  /api/health              health check
  POST /api/tts                 mp3 audio (OpenAI TTS, default voice "shimmer")
  POST /api/generate-question   fresh kid-friendly question (gpt-4o-mini)
  GET  /api/usage               returns the device's plan + remaining quota
  POST /api/checkout            creates a Stripe Checkout session
  POST /api/stripe-webhook      Stripe activates/cancels the subscription

Each request from the app sends an "X-Device-Id" header (UUID stored client-side
in localStorage). The backend tracks usage per device in a tiny SQLite db on the
Render persistent disk.

Plans:
  trial  — 7 AI question requests, then paywall (TTS free for retention).
  basic  — €2.99/month, 30 AI questions/month, refills each billing period.
  full   — €8.99/month or €69.99/year, unlimited.
"""

import io
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# --- Config -----------------------------------------------------------------
DB_PATH = os.environ.get("DB_PATH", "/var/data/magiko.db")

TRIAL_AI_LIMIT = 7
BASIC_AI_LIMIT = 30  # per Stripe billing period
PUBLIC_BASE_URL = os.environ.get(
    "PUBLIC_BASE_URL", "https://magiko-tetradio.onrender.com"
).rstrip("/")

PRICE_BASIC_MONTHLY = os.environ.get("PRICE_BASIC_MONTHLY", "").strip()
PRICE_FULL_MONTHLY  = os.environ.get("PRICE_FULL_MONTHLY", "").strip()
PRICE_FULL_YEARLY   = os.environ.get("PRICE_FULL_YEARLY", "").strip()
STRIPE_SECRET_KEY   = os.environ.get("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()

# --- App + CORS -------------------------------------------------------------
app = FastAPI(title="Magikó Tetrádio backend", version="2.0.0")

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
    expose_headers=["X-Device-Id"],
)

# --- SQLite -----------------------------------------------------------------
def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS devices (
                id                     TEXT PRIMARY KEY,
                plan                   TEXT NOT NULL DEFAULT 'trial',
                plan_status            TEXT NOT NULL DEFAULT 'active',
                ai_uses                INTEGER NOT NULL DEFAULT 0,
                ai_period_start        INTEGER NOT NULL DEFAULT 0,
                current_period_end     INTEGER NOT NULL DEFAULT 0,
                stripe_customer_id     TEXT DEFAULT '',
                stripe_subscription_id TEXT DEFAULT '',
                created_at             INTEGER NOT NULL,
                updated_at             INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_devices_sub ON devices(stripe_subscription_id);
            CREATE INDEX IF NOT EXISTS idx_devices_cus ON devices(stripe_customer_id);
            """
        )
_init_db()


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> int:
    return int(time.time())


def get_or_create_device(device_id: str) -> dict:
    if not device_id or len(device_id) < 8 or len(device_id) > 80:
        raise HTTPException(status_code=400, detail="Invalid device id")
    now = _now()
    with db() as con:
        row = con.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
        if row:
            return dict(row)
        con.execute(
            "INSERT INTO devices (id, created_at, updated_at) VALUES (?,?,?)",
            (device_id, now, now),
        )
        row = con.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
        return dict(row)


def _refresh_billing_period(dev: dict) -> dict:
    """If the user is on basic and the billing period has rolled, reset ai_uses."""
    if dev.get("plan") == "basic" and dev.get("ai_period_start"):
        # If current_period_end exists and is past, reset counter for the new period.
        cpe = int(dev.get("current_period_end") or 0)
        if cpe and _now() >= cpe:
            with db() as con:
                con.execute(
                    "UPDATE devices SET ai_uses=0, ai_period_start=?, updated_at=? WHERE id=?",
                    (_now(), _now(), dev["id"]),
                )
            dev["ai_uses"] = 0
            dev["ai_period_start"] = _now()
    return dev


def _ai_quota_for(plan: str) -> Optional[int]:
    if plan == "full":
        return None  # unlimited
    if plan == "basic":
        return BASIC_AI_LIMIT
    return TRIAL_AI_LIMIT  # trial


def _plan_status_dict(dev: dict) -> dict:
    plan = dev.get("plan") or "trial"
    status = dev.get("plan_status") or "active"
    is_active = status == "active"
    is_full = (plan == "full") and is_active
    quota = _ai_quota_for(plan if is_active else "trial")
    used = int(dev.get("ai_uses") or 0)
    remaining = None if quota is None else max(0, quota - used)
    return {
        "device_id": dev["id"],
        "plan": plan,
        "plan_status": status,
        "is_full": is_full,
        "is_active": is_active,
        "ai_used": used,
        "ai_quota": quota,                # None = unlimited
        "ai_remaining": remaining,        # None = unlimited
        "current_period_end": int(dev.get("current_period_end") or 0),
    }


def _consume_ai(device_id: str) -> dict:
    """Atomically check quota and increment. Raises 402 if no quota left."""
    with db() as con:
        row = con.execute("SELECT * FROM devices WHERE id=?", (device_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Unknown device")
        dev = dict(row)
        dev = _refresh_billing_period(dev)
        plan = dev.get("plan") or "trial"
        is_active = (dev.get("plan_status") or "active") == "active"
        effective_plan = plan if is_active else "trial"
        quota = _ai_quota_for(effective_plan)
        used = int(dev.get("ai_uses") or 0)
        if quota is not None and used >= quota:
            raise HTTPException(status_code=402, detail="quota_exceeded")
        con.execute(
            "UPDATE devices SET ai_uses=?, updated_at=? WHERE id=?",
            (used + 1, _now(), device_id),
        )
        dev["ai_uses"] = used + 1
        return dev


def _require_device(x_device_id: Optional[str]) -> dict:
    if not x_device_id:
        raise HTTPException(status_code=400, detail="X-Device-Id header required")
    return get_or_create_device(x_device_id)


# --- OpenAI client ----------------------------------------------------------
_oa: Optional[OpenAI] = None


def openai_client() -> OpenAI:
    global _oa
    if _oa is None:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise HTTPException(status_code=503, detail="OpenAI not configured")
        _oa = OpenAI(api_key=key)
    return _oa


# --- Stripe client ----------------------------------------------------------
_stripe = None


def stripe_module():
    global _stripe
    if _stripe is None:
        if not STRIPE_SECRET_KEY:
            raise HTTPException(status_code=503, detail="Stripe not configured")
        import stripe as _s
        _s.api_key = STRIPE_SECRET_KEY
        _stripe = _s
    return _stripe


PLAN_FROM_PRICE = {
    PRICE_BASIC_MONTHLY: "basic",
    PRICE_FULL_MONTHLY:  "full",
    PRICE_FULL_YEARLY:   "full",
}


# ============================================================================
# Endpoints
# ============================================================================
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "magiko-tetradio-backend",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "prices_configured": all([PRICE_BASIC_MONTHLY, PRICE_FULL_MONTHLY, PRICE_FULL_YEARLY]),
    }


@app.get("/api/usage")
def usage(x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    dev = _require_device(x_device_id)
    dev = _refresh_billing_period(dev)
    return _plan_status_dict(dev)


# --- TTS --------------------------------------------------------------------
ALLOWED_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
DEFAULT_VOICE = "shimmer"
MAX_TTS_CHARS = 800


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None


@app.post("/api/tts")
def tts(req: TTSRequest, x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    # We don't gate TTS — voice is fundamental UX, kept open even on trial.
    if x_device_id:
        try: _require_device(x_device_id)
        except HTTPException: pass

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS]

    voice = (req.voice or DEFAULT_VOICE).strip().lower()
    if voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    speed = req.speed if req.speed is not None else 0.9
    try: speed = float(speed)
    except (TypeError, ValueError): speed = 0.9
    speed = max(0.5, min(1.5, speed))

    try:
        response = openai_client().audio.speech.create(
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
        print(f"[tts] error: {exc!r}")
        raise HTTPException(status_code=502, detail=f"TTS error: {type(exc).__name__}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


# --- Generate question -------------------------------------------------------
MAX_TOPIC_CHARS = 120
MAX_AVOID_ITEMS = 8
MAX_AVOID_CHAR_PER_ITEM = 200

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
    module: Optional[str] = None
    moduleTitle: Optional[str] = None
    grade: Optional[str] = None
    level: Optional[str] = "normal"
    topic: Optional[str] = None
    avoid: Optional[list[str]] = None


def _sanitize_avoid(items: Optional[list[str]]) -> list[str]:
    if not items: return []
    out: list[str] = []
    for it in items[:MAX_AVOID_ITEMS]:
        if not isinstance(it, str): continue
        s = it.strip()
        if not s: continue
        if len(s) > MAX_AVOID_CHAR_PER_ITEM: s = s[:MAX_AVOID_CHAR_PER_ITEM]
        out.append(s)
    return out


SYSTEM_PROMPT = (
    "Είσαι σχεδιάστρια εκπαιδευτικού περιεχομένου για ελληνική παιδική εφαρμογή.\n"
    "Δημιουργείς ΜΙΑ ερώτηση πολλαπλής επιλογής, στα Ελληνικά, ασφαλή και χαρούμενη.\n\n"
    "Κανόνες:\n"
    "- Όλα τα κείμενα στα ΕΛΛΗΝΙΚΑ.\n"
    "- Ακριβώς 4 επιλογές απάντησης, μία σωστή. Καμία επιλογή δεν επαναλαμβάνεται.\n"
    "- Πάντα κατάλληλο για παιδί της δηλωμένης τάξης/ηλικίας.\n"
    "- Σύντομη, καθαρή διατύπωση. Χωρίς ειρωνεία ή φόβο.\n"
    "- ΠΟΤΕ να μην επαναλάβεις μία από τις ερωτήσεις στο πεδίο \"avoid\".\n"
    "- Απάντησε ΜΟΝΟ με έγκυρο JSON ακριβώς της παρακάτω δομής."
)
JSON_SHAPE = (
    '{\n'
    '  "question": "Η ερώτηση στα Ελληνικά (μία πρόταση).",\n'
    '  "emoji": "1-3 emojis που εικονογραφούν (μπορεί να είναι κενό)",\n'
    '  "answers": ["επιλογή1", "επιλογή2", "επιλογή3", "επιλογή4"],\n'
    '  "correct": 0,\n'
    '  "help": "Σύντομη μία πρόταση εξήγηση γιατί η σωστή είναι σωστή.",\n'
    '  "tag": "μία λέξη/φράση κατηγορίας"\n'
    '}'
)


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try: return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m: return json.loads(m.group(0))
        raise


def _validate_question(obj: dict) -> dict:
    if not isinstance(obj, dict): raise ValueError("not an object")
    q = (obj.get("question") or "").strip()
    answers = obj.get("answers") or []
    correct = obj.get("correct")
    if not q: raise ValueError("missing question")
    if not isinstance(answers, list) or len(answers) != 4:
        raise ValueError("answers must be exactly 4")
    cleaned = []
    seen = set()
    for a in answers:
        s = str(a).strip()
        if not s or s in seen: raise ValueError("answers must be unique and non-empty")
        seen.add(s); cleaned.append(s)
    if not isinstance(correct, int) or correct < 0 or correct > 3:
        raise ValueError("correct must be integer 0..3")
    return {
        "question": q,
        "emoji": str(obj.get("emoji") or "").strip(),
        "answers": cleaned,
        "correct": correct,
        "help": str(obj.get("help") or "").strip(),
        "tag": (str(obj.get("tag") or "").strip() or "γενική")[:40],
    }


@app.post("/api/generate-question")
def generate_question(
    req: GenQuestionRequest,
    x_device_id: Optional[str] = Header(None, alias="X-Device-Id"),
):
    # Quota check (+ atomic increment) BEFORE we burn an OpenAI call
    _require_device(x_device_id)
    dev_after = _consume_ai(x_device_id)  # raises 402 if exhausted

    module = (req.module or "").strip().lower()
    module_hint = MODULE_HINTS.get(module, "")
    title = (req.moduleTitle or "").strip()
    grade_hint = GRADE_HINTS.get((req.grade or "").strip().lower(), "")
    level_hint = LEVEL_HINTS.get((req.level or "normal").strip().lower(), LEVEL_HINTS["normal"])
    topic = (req.topic or "").strip()
    if len(topic) > MAX_TOPIC_CHARS: topic = topic[:MAX_TOPIC_CHARS]
    avoid = _sanitize_avoid(req.avoid)
    avoid_block = ("\nΕρωτήσεις που ΔΕΝ πρέπει να επαναληφθούν (avoid):\n" +
                   "\n".join(f"- {a}" for a in avoid)) if avoid else ""

    parts = []
    if title:        parts.append(f"Θεματική ενότητα: {title}.")
    if module_hint:  parts.append(f"Λεπτομέρειες θεματικής: {module_hint}")
    if grade_hint:   parts.append(f"Επίπεδο τάξης: {grade_hint}")
    parts.append(f"Δυσκολία: {level_hint}")
    if topic:        parts.append(f"Ειδικό αίτημα γονέα: «{topic}». Αν ταιριάζει στη θεματική, δώσε προτεραιότητα.")

    user_prompt = (
        "Δημιούργησε ΜΙΑ νέα ερώτηση πολλαπλής επιλογής για το παιδί.\n\n"
        + "\n".join(parts)
        + "\n\nΔομή JSON (μόνο JSON, χωρίς σχόλια):\n" + JSON_SHAPE
        + avoid_block
    )

    try:
        completion = openai_client().chat.completions.create(
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
        out = _validate_question(data)
        # also include the updated usage so the UI updates instantly
        out["usage"] = _plan_status_dict(dev_after)
        return out
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[generate-question] error: {exc!r}")
        # The AI use was consumed but we couldn't satisfy it — refund it.
        try:
            with db() as con:
                con.execute(
                    "UPDATE devices SET ai_uses=MAX(0, ai_uses-1), updated_at=? WHERE id=?",
                    (_now(), x_device_id),
                )
        except Exception: pass
        raise HTTPException(status_code=502, detail=f"OpenAI: {type(exc).__name__}: {str(exc)[:280]}")


# --- Stripe checkout --------------------------------------------------------
class CheckoutRequest(BaseModel):
    plan: str  # "basic_monthly" | "full_monthly" | "full_yearly"


PRICE_LOOKUP = {
    "basic_monthly": "PRICE_BASIC_MONTHLY",
    "full_monthly":  "PRICE_FULL_MONTHLY",
    "full_yearly":   "PRICE_FULL_YEARLY",
}


@app.post("/api/checkout")
def checkout(req: CheckoutRequest,
             x_device_id: Optional[str] = Header(None, alias="X-Device-Id")):
    dev = _require_device(x_device_id)
    key = PRICE_LOOKUP.get(req.plan)
    if not key:
        raise HTTPException(status_code=400, detail="Unknown plan")
    price_id = {
        "PRICE_BASIC_MONTHLY": PRICE_BASIC_MONTHLY,
        "PRICE_FULL_MONTHLY":  PRICE_FULL_MONTHLY,
        "PRICE_FULL_YEARLY":   PRICE_FULL_YEARLY,
    }[key]
    if not price_id:
        raise HTTPException(status_code=503, detail=f"{req.plan} not configured")

    try:
        s = stripe_module()
        session_kwargs = dict(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{PUBLIC_BASE_URL}/?checkout=success&plan={req.plan}",
            cancel_url=f"{PUBLIC_BASE_URL}/?checkout=cancel",
            client_reference_id=dev["id"],
            metadata={"device_id": dev["id"], "plan": req.plan},
            allow_promotion_codes=True,
        )
        if dev.get("stripe_customer_id"):
            session_kwargs["customer"] = dev["stripe_customer_id"]
        session = s.checkout.Session.create(**session_kwargs)
        return {"url": session.url}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[checkout] error: {exc!r}")
        raise HTTPException(status_code=502, detail=f"Stripe: {type(exc).__name__}: {str(exc)[:240]}")


# --- Stripe webhook ---------------------------------------------------------
@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="webhook secret not configured")
    s = stripe_module()
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = s.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad signature: {exc}")

    etype = event["type"]
    obj   = event["data"]["object"]

    def _activate(device_id: str, sub):
        plan_name = "basic"
        # Identify which price was used
        try:
            items = sub.get("items", {}).get("data", []) if isinstance(sub, dict) else []
            if items:
                price_id = items[0].get("price", {}).get("id", "")
                plan_name = PLAN_FROM_PRICE.get(price_id, "basic")
        except Exception: pass
        cpe = int(sub.get("current_period_end") or 0) if isinstance(sub, dict) else 0
        with db() as con:
            con.execute(
                """UPDATE devices SET
                       plan=?, plan_status='active',
                       current_period_end=?, ai_uses=0, ai_period_start=?,
                       stripe_customer_id=?, stripe_subscription_id=?,
                       updated_at=?
                   WHERE id=?""",
                (plan_name, cpe, _now(),
                 (sub.get("customer") if isinstance(sub, dict) else "") or "",
                 (sub.get("id") if isinstance(sub, dict) else "") or "",
                 _now(), device_id),
            )

    def _deactivate_by_sub(subscription_id: str):
        if not subscription_id: return
        with db() as con:
            con.execute(
                """UPDATE devices SET plan='trial', plan_status='canceled', updated_at=?
                   WHERE stripe_subscription_id=?""",
                (_now(), subscription_id),
            )

    try:
        if etype == "checkout.session.completed":
            device_id = (obj.get("metadata") or {}).get("device_id") or obj.get("client_reference_id") or ""
            sub_id = obj.get("subscription") or ""
            if device_id and sub_id:
                sub = s.Subscription.retrieve(sub_id)
                _activate(device_id, sub if isinstance(sub, dict) else dict(sub))
        elif etype in ("customer.subscription.created", "customer.subscription.updated"):
            sub = obj
            sub_id = sub.get("id") or ""
            device_id = (sub.get("metadata") or {}).get("device_id") or ""
            # If we don't have device_id in metadata, try to match by sub id
            if not device_id and sub_id:
                with db() as con:
                    row = con.execute(
                        "SELECT id FROM devices WHERE stripe_subscription_id=?",
                        (sub_id,),
                    ).fetchone()
                    if row: device_id = row["id"]
            if device_id:
                status = sub.get("status", "")
                if status in ("active", "trialing"):
                    _activate(device_id, sub)
                else:
                    _deactivate_by_sub(sub_id)
        elif etype == "customer.subscription.deleted":
            _deactivate_by_sub(obj.get("id") or "")
    except Exception as exc:  # noqa: BLE001
        print(f"[webhook {etype}] error: {exc!r}")

    return {"received": True}
