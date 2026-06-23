from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

_embed_model = None
_reranker = None
_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-m3")
    return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading reranker model...")
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="./storage")
        _collection = client.get_collection("papers")
    return _collection


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    embed_model = _get_embed_model()
    collection = _get_collection()

    q_embedding = embed_model.encode(question)

    # Retrieve 20 candidates, then rerank to top_k
    results = collection.query(
        query_embeddings=[q_embedding.tolist()],
        n_results=20
    )

    candidates = []
    for i in range(len(results["documents"][0])):
        candidates.append({
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "paper_id": results["metadatas"][0][i]["paper_id"],
        })

    reranker = _get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]
