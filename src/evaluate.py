from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

TEST_SET = [
    # (问题, 正确来源论文的 nickname)
    ("What noise schedule does DDPM use during the forward process?", "DDPM"),
    ("How does DDPM define the reverse diffusion process?", "DDPM"),
    ("What is the training objective of DDPM?", "DDPM"),
    ("How does DDIM accelerate sampling compared to DDPM?", "DDIM"),
    ("What makes DDIM a non-Markovian process?", "DDIM"),
    ("How does DDIM achieve deterministic sampling?", "DDIM"),
    ("What is classifier-free guidance and how does it work?", "CFG"),
    ("How does guidance strength affect image generation quality?", "CFG"),
    ("What is the latent diffusion model in Stable Diffusion?", "Stable Diffusion"),
    ("How does Stable Diffusion use a VAE to compress images?", "Stable Diffusion"),
    ("How does Zero123 control the camera viewpoint?", "Zero123"),
    ("What dataset does Zero123 use for training?", "Zero123"),
    ("How does Zero123++ improve over Zero123?", "Zero123++"),
    ("What is the multi-view consistency approach in Zero123++?", "Zero123++"),
    ("How does DreamFusion use score distillation sampling?", "DreamFusion"),
    ("What is the role of NeRF in DreamFusion?", "DreamFusion"),
    ("How does Magic3D achieve higher resolution than DreamFusion?", "Magic3D"),
    ("What two-stage approach does Magic3D use?", "Magic3D"),
    ("How does Point-E generate 3D point clouds from text?", "Point-E"),
    ("What is the clip-guided diffusion model in Point-E?", "Point-E"),
]

NICKNAME_TO_TITLE = {
    "DDPM": "Denoising Diffusion Probabilistic Models",
    "DDIM": "Denoising Diffusion Implicit Models",
    "CFG": "Classifier-Free Diffusion Guidance",
    "Stable Diffusion": "High-Resolution Image Synthesis with Latent Diffusion Models",
    "Zero123": "Zero-1-to-3: Zero-shot One Image to 3D Object",
    "Zero123++": "Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model",
    "DreamFusion": "DreamFusion: Text-to-3D using 2D Diffusion",
    "Magic3D": "Magic3D: High-Resolution Text-to-3D Content Creation",
    "Point-E": "Point-E: A System for Generating 3D Point Clouds from Complex Prompts",
}


def evaluate(use_reranker: bool, top_k: int = 5) -> float:
    embed_model = SentenceTransformer("BAAI/bge-m3")
    client = chromadb.PersistentClient(path="./storage")
    collection = client.get_collection("papers")

    reranker = None
    if use_reranker:
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    hits = 0
    for question, correct_nickname in TEST_SET:
        correct_title = NICKNAME_TO_TITLE[correct_nickname]

        q_embedding = embed_model.encode(question)
        results = collection.query(
            query_embeddings=[q_embedding.tolist()],
            n_results=20 if use_reranker else top_k
        )

        candidates = []
        for i in range(len(results["documents"][0])):
            candidates.append({
                "text": results["documents"][0][i],
                "title": results["metadatas"][0][i]["title"],
            })

        if use_reranker:
            pairs = [(question, c["text"]) for c in candidates]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            top_chunks = [c for _, c in ranked[:top_k]]
        else:
            top_chunks = candidates[:top_k]

        retrieved_titles = {c["title"] for c in top_chunks}
        if correct_title in retrieved_titles:
            hits += 1

    return hits / len(TEST_SET)


if __name__ == "__main__":
    print("评估中（不加 Reranker）...")
    score_base = evaluate(use_reranker=False)

    print("评估中（加 Reranker）...")
    score_rerank = evaluate(use_reranker=True)

    print("\n========== 评估结果 ==========")
    print(f"Hit Rate@5（不加 Reranker）: {score_base:.1%}")
    print(f"Hit Rate@5（加 Reranker）:   {score_rerank:.1%}")
    print(f"提升:                        +{(score_rerank - score_base):.1%}")
    print("================================")
