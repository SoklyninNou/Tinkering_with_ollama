import time
import os

def load_context(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def transcript(file_path, speaker, text):
    with open(file_path, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {speaker}: {text}\n")
        
# Putting all lecture .typ files into one file
def group_lecture(directory_path, output_file):
    lecture_counter = 1
    with open(os.path.join(directory_path, output_file), 'w', encoding='utf-8') as f:
            f.write("Lecture Contents:\n")
    for file in os.listdir(directory_path):
        if file.endswith('.typ'):
            with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
                content = f.read()
        with open(os.path.join(directory_path, output_file), 'a', encoding='utf-8') as f:
            f.write(f"\n--- Lecture {lecture_counter} ---\n")
            lecture_counter += 1
            f.write(content)
            f.write("\n")

# Mass renaming files in a directory
def mass_renaming(directory_path, file_format):
    counter = 1
    for file in os.listdir(directory_path):
        if file.endswith(file_format):
            new_name = f"lecture_{counter}.{file_format}"
            os.rename(os.path.join(directory_path, file), os.path.join(directory_path, new_name))
            counter += 1
            
mass_renaming("data/lecture-image", "jpg")
# group_lecture("data/lectures", "grouped_lectures.txt")