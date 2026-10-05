# src/rag/generate.py
import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import retrieve

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

SYSTEM_PROMPT = """You are a question-answering assistant. Answer the question
using ONLY the information in the provided context below. Do not use any
outside knowledge. If the context does not contain the answer, respond
exactly with: "I don't have enough information to answer that."
Cite which source document(s) you used in your answer."""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(f"[Source: {chunk['doc']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, n_results: int = 3) -> dict:
    retrieved_chunks = retrieve(question, n_results=n_results)
    prompt = build_prompt(question, retrieved_chunks)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    answer = response.json().get("response", "").strip()

    return {
        "question": question,
        "answer": answer,
        "sources": [c["doc"] for c in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    result = generate_answer("When did Alan Turing die?")
    print("Question:", result["question"])
    print("\nAnswer:", result["answer"])
    print("\nSources used:", set(result["sources"]))