from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


class Node:
    def __init__(self, title: str, level: int):
        self.title = title
        self.level = level
        self.children: List["Node"] = []
        self.content: List[str] = []
        self.embedding = None

    def add_child(self, node: "Node"):
        self.children.append(node)

    def add_content(self, line: str):
        self.content.append(line)

    def __repr__(self):
        return f"Node(title={self.title!r}, level={self.level})"


def build_tree(lines: List[str]) -> Node:
    root = Node("ROOT", 0)
    stack = [root]
    current_node = root

    for line in lines:
        raw = line.rstrip()

        # Header line
        if raw.lstrip().startswith("="):
            level = len(raw) - len(raw.lstrip("="))
            title = raw[level:].strip()

            node = Node(title, level)

            while stack and stack[-1].level >= level:
                stack.pop()

            stack[-1].add_child(node)
            stack.append(node)
            current_node = node

        # Content line
        else:
            if current_node is not None:
                current_node.add_content(raw)

    return root

def dict_node(node: Node):
    if node is None:
        return None
    serialized_node = {
        node.embedding:
        [
            node.title,
            node.level,
            node.children,
            node.content
        ]
        }
    
def save_node(node: Node, tree_file):
    with open(tree_file, "w", encoding="utf-8") as f:
        f.write(dict_node(node))
    
def compute_embeddings(node: Node):
    if node.title != "ROOT":
        text = node.title + " " + " ".join(node.content[:5])
        node.embedding = model.encode(text, convert_to_tensor=True)

    for child in node.children:
        compute_embeddings(child)


def find_k_most_relevant_sections(
    root: Node, prompt: str, k: int = 6
) -> List[Tuple[Node, float]]:
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)

    results: List[Tuple[Node, float]] = []
    seen = set()
    
    def dfs(node: Node):
        if node.embedding is not None:
            if node.title not in seen:
                seen.add(node.title)
                score = float(util.cos_sim(prompt_embedding, node.embedding))
                results.append((node, score))
        for child in node.children:
            dfs(child)

    dfs(root)

    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:k]


def print_section(node: Node):
    print(f"\n{'=' * node.level} {node.title}")
    for line in node.content:
        print(line)


def print_tree(node: Node, indent: int = 0):
    if node.title != "ROOT":
        print("  " * indent + node.title)
    for child in node.children:
        print_tree(child, indent + 1)

def print_titles(sections: Tuple):
    titles = []
    for node, _ in sections:
        print("    - " + node.title)

if __name__ == "__main__":
    with open("data/grouped_lectures.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    tree = build_tree(lines)
    compute_embeddings(tree)
    save_node(Node("ROOT", 0), "data/tree.txt")

#     prompt = "What is k-consistency"

#     sections = find_k_most_relevant_sections(tree, prompt, k=6)

#     print(f"\nPrompt: {prompt}\n")
#     print("Document structure:")
#     print_tree(tree)

#     print("\nTop relevant sections:")
#     for node, score in sections:
#         print(f"\nScore: {score:.4f}")
#         print_section(node)

