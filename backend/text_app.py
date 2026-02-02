from agent import generate_response
from helper import transcript
import time

file_path = "./data/transcript.txt"
with open(file_path, 'a') as f:
    f.write(f"\nNew Conversation Started: [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

def generate_text_response():
    while True:
        input_text = input("You: ")
        transcript(file_path, "You", input_text)
        if input_text.lower() in ['exit', 'quit']:
            transcript(file_path, "Conversation Ended", "\n")
            break
        if input_text.lower() in ['goodbye', 'bye', 'see you']:
            generate_response(file_path, "goodbye")
            break
        print("Slybot: ", end='', flush=True)
        response = generate_response(file_path, input_text)
        print()
        transcript(file_path, "Slybot", response)
        
if __name__ == "__main__":
    generate_text_response()