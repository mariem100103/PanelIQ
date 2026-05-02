import os
import whisper
import subprocess
from groq import Groq


from pathlib import Path
env = Path('C:/Users/user/Desktop/quizlive/.env').read_text()
for line in env.splitlines():
    if '=' in line:
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

client = Groq(api_key=os.environ['GROQ_API_KEY'])


subprocess.run([
    'ffmpeg', '-i',
    'C:/Users/user/Desktop/quizlive/videos/fsf_ep36.mp4',
    '-ss', '00:00:30', '-t', '60',
    '-vn', '-ar', '16000', '-ac', '1',
    'C:/Users/user/Desktop/quizlive/videos/agent_audio.wav',
    '-y'
], capture_output=True)


print("Transcription...")
model = whisper.load_model("medium")
result = model.transcribe(
    'C:/Users/user/Desktop/quizlive/videos/agent_audio.wav',
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