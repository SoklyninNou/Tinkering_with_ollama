import time

def load_context(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def transcript(file_path, speaker, text):
    with open(file_path, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {speaker}: {text}\n")