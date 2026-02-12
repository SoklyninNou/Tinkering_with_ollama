import time
import ollama
from helper import *
from section_tree import *
from tools import *

tools = {
    "write_to_file": {
        "func": reminder,
        "description": (
            "Writes the date, hour, and minute of the reminder."
            "The tuple is (int, bool) should be the time and whether of not it is relative"
            "Set the boolean to true if the prompt contains relative time such as: in X minutes, in X hour,tommorrow, or in X days."
            "Set the boolean to false if the prompt contains absolute time such as: at X time or on X date."
            "Input should be a JSON object with day: tuple, hour: tuple, minute: tuple, and remind: str."
        )
    },
}

def generate_response(file_path, user_prompt):
    transcript_text = load_context(file_path)
    
    # Relevant Lecture Extraction
    with open("data/grouped_lectures.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    tree = build_tree(lines)
    compute_embeddings(tree)
    relevant_lectures = find_k_most_relevant_sections(tree, user_prompt)

    initial_sys_message = load_context("./data/test.txt")

    # system_message = initial_sys_message.format(
    #     lectures=relevant_lectures[1],
    #     transcript=transcript_text
    # )initial_sys_message

    # Call the model
    response = ollama.chat(
        model='qwen3',
        messages=[
            {'role': 'system', 'content': initial_sys_message},
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

# Tool call output
# {
#   "completion_message": {
#     "content": {
#       "type": "text",
#       "text": ""
#     },
#     "role": "assistant",
#     "stop_reason": "tool_calls",
#     "tool_calls": [
#       {
#         "id": "466d49b7-8641-43bd-844e-ecac6a818974",
#         "function": {
#           "name": "get_weather",
#           "arguments": "{\"location\":\"Menlo Park\"}"
#         }
#       }
#     ]
#   }
# }