import ollama
from helper import load_context
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaProcessor

def generate_response(file_path, user_prompt):
    transcript = load_context(file_path)
    lectures = "./data/grouped_lectures.txt"

    initial_sys_message = load_context("./data/system_message.txt")
    system_message = initial_sys_message.format(
        lectures=lectures,
        transcript=transcript
    )   
    
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': system_message
            },
            {
                'role': 'user',
                'content': user_prompt,
            }
        ],
        stream=True
    )
    
    buffer = ""
    full_response = ""
    for chunk in response:
        content = chunk['message']['content']
        buffer += content
        full_response += content
        
        if ' ' in buffer:
            words = buffer.split(' ')
            for word in words[:-1]:
                print(word, end=' ', flush=True)
            buffer = words[-1]
    return full_response

# while True:
#     prompt = input("Enter your prompt: ")
#     if prompt.lower() in ['exit', 'quit']:
#         break
#     print("Slybot: ", end='', flush=True)
#     generate_response(prompt)

