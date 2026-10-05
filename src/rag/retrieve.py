# src/rag/retrieve.py
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None
_client = None
_collection = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, n_results: int = 3) -> list[dict]:
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    retrieved = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({"text": doc, "doc": meta["doc"], "distance": distance})
    return retrieved


if __name__ == "__main__":
    results = retrieve("When did Alan Turing die?")
    for r in results:
        print(f"[{r['doc']}] (distance={r['distance']:.4f})")
        print(r["text"][:150])
        print()