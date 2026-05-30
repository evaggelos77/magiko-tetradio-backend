# Μαγικό Τετράδιο AI Junior — backend

Μικρό FastAPI backend που εξυπηρετεί την παιδική εφαρμογή
[magiko-tetradio.onrender.com](https://magiko-tetradio.onrender.com/) με
**premium γυναικεία Ελληνική φωνή** μέσω του OpenAI TTS, ώστε ο ήχος να
ακούγεται σταθερά καθαρός σε κάθε συσκευή.

## Endpoints

- `GET /api/health` — health check
- `POST /api/tts` — body `{ "text": "...", "voice": "shimmer" }`, επιστρέφει
  audio/mpeg (mp3). Default φωνή: `shimmer` (απαλή, γυναικεία).

## Deploy

Static config στο `render.yaml`. Στο Render dashboard πρέπει να οριστεί το
`OPENAI_API_KEY` (sync:false — μυστικό).

— μια εφαρμογή της **ev labs ai** · evlabsai.gr
