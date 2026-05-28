from sentence_transformers import SentenceTransformer
import chromadb

_embed_model = None
_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-m3")
    return _embed_model


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

    results = collection.query(
        query_embeddings=[q_embedding.tolist()],
        n_results=top_k
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "paper_id": results["metadatas"][0][i]["paper_id"],
        })

    return chunks
