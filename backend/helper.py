import time
import os
from pathlib import Path
from bs4 import BeautifulSoup

def load_context(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def transcript(file_path, speaker, text):      
    with open(file_path, 'a') as f:
        if speaker == "":
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
        else:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {speaker}: {text}\n")
        
# Putting all lecture files into one file
def group_lecture(directory_path, output_file):
    lecture_counter = 1
    with open(os.path.join(directory_path, output_file), 'w', encoding='utf-8') as f:
            f.write("Lecture Contents:\n")
    for file in os.listdir(directory_path):
        if file.endswith('.txt'):
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

def html_to_txt():
    HTML_DIR = Path("./data/lectures")
    TXT_DIR = Path("./data/lectures_txt")
    TXT_DIR.mkdir(exist_ok=True)

    for html_file in sorted(HTML_DIR.glob("lecture_*.html")):
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        text = soup.get_text(separator="\n")

        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        txt_file = TXT_DIR / (html_file.stem + ".txt")
        txt_file.write_text(text, encoding="utf-8")
    
# html_to_txt()
# mass_renaming("data/lecture-image", "jpg")
# group_lecture("data/lectures", "grouped_lectures.txt")