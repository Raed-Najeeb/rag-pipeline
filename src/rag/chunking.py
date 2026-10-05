from pathlib import Path

def chunk_text(text:str, chunk_size: int=500, overlap: int=50) -> list[str]: 
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_document(filepath:str, chunk_size: int=500, overlap: int=50) -> list[dict] :
    text = Path(filepath).read_text(encoding="utf-8")
    raw_chunks = chunk_text(text, chunk_size, overlap)
    doc_name = Path(filepath).stem
    return [
        {"doc":doc_name, "chunk_id": f"{doc_name}_{i}", "text": chunk}
        for i, chunk in enumerate(raw_chunks)
    ]

if __name__ == "__main__" :
    import json

    all_chunks = []
    for filepath in Path("data/documents").glob("*.txt"):
        chunks = chunk_document(str(filepath))
        all_chunks.extend(chunks)
        print(f"{filepath.name}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open("data/results/chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")