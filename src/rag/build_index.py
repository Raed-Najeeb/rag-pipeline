import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/results/chunks.jsonl"
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_chunks(path:str) -> list[dict] :
    chunks = []
    with open(path, encoding = "utf-8") as f :
        for line in f :
            chunks.append(json.loads(line))
    return chunks

def main() :
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"doc": c["doc"]} for c in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar = True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        ids = ids,
        embeddings = embeddings, 
        documents = texts,
        metadatas = metadatas,
    )

    print(f"\nStored {collection.count()} chunks in ChromaDB at {CHROMA_DB_PATH}")

if __name__ == "__main__" :
    main()