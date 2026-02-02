import ollama
from helper import load_context

def generate_response(file_path, user_prompt):
    transcript = load_context(file_path)
    system_message = (
        "You are a student named SlyBot that assists users in general tasks.\n"
        # "Never mention that you are an AI model.\n"
        "The transcript of the conversation is in the following format:\n"
        "[year-month-day hour:minute:second] <text>\n"
        "The time is provided for context only and should not be included in your responses.\n"
        "It is the closest approximation of current.\n"
        "Use this transcript to understand context and respond appropriately but don't output any timestamps.\n"
        "Transcript:\n"
        f"{transcript}"
    )
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_prompt}
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

