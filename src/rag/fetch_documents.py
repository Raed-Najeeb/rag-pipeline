import wikipediaapi
import os

wiki = wikipediaapi.Wikipedia(user_agent="rag-pipeline-project", language="en")

topics = [
    "Alan Turing",
    "History of artificial intelligence",
    "Transformer (deep learning architecture)",
    "Neural network",
    "Machine learning",
]

os.makedirs("data/documents", exist_ok=True)

for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        filename = topic.replace(" ", "_").replace("(", "").replace(")", "") + ".txt"
        filepath = os.path.join("data/documents", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page.text)
        print(f"Saved {filepath} ({len(page.text)} characters)")
    else:
        print(f"Page not found: {topic}")