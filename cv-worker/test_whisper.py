import whisper
import subprocess

# Extraire 30 secondes d'audio de la vidéo
subprocess.run([
    'ffmpeg', '-i', 
    'C:/Users/user/Desktop/quizlive/videos/fsf_ep36.mp4',
    '-ss', '00:05:00',
    '-t', '30',
    '-vn', '-ar', '16000', '-ac', '1',
    'C:/Users/user/Desktop/quizlive/videos/test_audio.wav',
    '-y'
])

print("Audio extrait, chargement de Whisper...")
model = whisper.load_model("medium")

print("Transcription en cours...")
result = model.transcribe(
    'C:/Users/user/Desktop/quizlive/videos/test_audio.wav',
    language='ar'
)

print("\n--- TRANSCRIPTION ---")
print(result['text'])