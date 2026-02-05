import time
import ollama
from helper import *
from section_tree import *

def generate_response(file_path, user_prompt):
    transcript_text = load_context(file_path)
    
    # Relevant Lecture Extraction
    with open("data/grouped_lectures.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    tree = build_tree(lines)
    compute_embeddings(tree)
    relevant_lectures = find_k_most_relevant_sections(tree, user_prompt)

    initial_sys_message = load_context("./data/system_message.txt")

    system_message = initial_sys_message.format(
        lectures=relevant_lectures[1],
        transcript=transcript_text
    )

    # Call the model
    response = ollama.chat(
        model='qwen3',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_prompt}
        ],
        stream=True
    )

    # Stream output
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

    if buffer:
        print(buffer, flush=True)
    print("Some relevant sections:")
    print_titles(relevant_lectures)
    return full_response
