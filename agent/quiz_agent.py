import os
import whisper
import subprocess
from groq import Groq

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=groq_key)

VIDEO_FILE = ROOT / "videos" / "fsf_ep36.mp4"
AUDIO_FILE = ROOT / "videos" / "agent_audio.wav"

subprocess.run([
    'ffmpeg', '-i',
    str(VIDEO_FILE),
    '-ss', '00:00:30', '-t', '60',
    '-vn', '-ar', '16000', '-ac', '1',
    str(AUDIO_FILE),
    '-y'
], capture_output=True)


print("Transcription...")
model = whisper.load_model("medium")
result = model.transcribe(
    str(AUDIO_FILE),
    language='ar'
)
transcript = result['text']
print(f"Transcription: {transcript}\n")


print("Génération question...")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": """Tu es un générateur de questions de quiz pour une émission tunisienne.
À partir d'une transcription audio, tu dois:
1. Extraire le nom de l'invité si mentionné
2. Générer UNE question QCM pertinente
Réponds UNIQUEMENT en JSON valide, rien d'autre:
{
  "guest_name": "nom ou null",
  "question": "la question en arabe",
  "options": ["option1", "option2", "option3", "option4"],
  "correct_index": 0,
  "category": "ترفيه/رياضة/سياسة",
  "confidence": 0.85
}"""
        },
        {
            "role": "user",
            "content": f"Transcription: {transcript}"
        }
    ]
)

import json
output = response.choices[0].message.content
print("\n--- QUESTION GÉNÉRÉE ---")
print(json.dumps(json.loads(output), ensure_ascii=False, indent=2))