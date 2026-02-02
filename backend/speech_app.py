from vosk import Model, KaldiRecognizer
import pyaudio
import json
from agent import generate_response
import numpy as np
import time
from helper import transcript

model = Model("vosk-model-en-us-0.42-gigaspeech")
rec = KaldiRecognizer(model, 16000)
SILENCE_THRESHOLD = 50
# Create or clear transcript file

file_path = "./data/transcript.txt"
with open(file_path, 'a') as f:
    f.write(f"New Conversation Started: [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000)
stream.start_stream()

print("Slybot Online.")

while True:
    data = stream.read(8000, exception_on_overflow=False)
    audio_chunk = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(audio_chunk)))

    if rec.AcceptWaveform(data):
        # Final result
        result = json.loads(rec.Result())
        text = result.get("text", "")
        if text.strip():
            transcript(file_path, "You", text)
            print()
            print("You (Full): " + text)
            print("Slybot: ", end='', flush=True)
            response = generate_response(text)
            transcript(file_path, "Slybot", response)
                
    else:
        # Partial result
        partial = json.loads(rec.PartialResult())
        text = partial.get("partial", "")
        if text.strip():
            print("You (Partial): " + text, end="\r")

