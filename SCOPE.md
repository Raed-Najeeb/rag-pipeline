# Scope

**What this project builds:** a retrieval-augmented generation (RAG) pipeline —
given a small set of documents and a question, retrieve the most relevant
chunk(s) and generate a grounded answer using a locally-served Llama 3.1 8B
(via Ollama).

**What this tests/demonstrates:**
1. Document chunking strategy
2. Embedding-based semantic retrieval (ChromaDB + sentence-transformers)
3. Grounded generation — answering strictly from retrieved context

**Out of scope:** fine-tuning the embedding model, multi-turn conversation,
re-ranking (cross-encoder re-ranking is a possible later extension, not core
scope), production-scale document volumes.