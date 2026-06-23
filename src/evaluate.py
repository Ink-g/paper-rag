from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

TEST_SET = [
    # (问题, 正确来源论文的 nickname)
    # 扩散模型基础
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
    # 扩散模型应用
    ("How does ControlNet add spatial conditioning to diffusion models?", "ControlNet"),
    ("What is the zero convolution technique in ControlNet?", "ControlNet"),
    ("How does InstructPix2Pix edit images using text instructions?", "InstructPix2Pix"),
    ("What dataset is used to train InstructPix2Pix?", "InstructPix2Pix"),
    ("What text encoder does Imagen use and why?", "Imagen"),
    ("How does Imagen achieve photorealistic text-to-image generation?", "Imagen"),
    ("How does SDXL improve over the original Stable Diffusion?", "SDXL"),
    ("What is the two-stage pipeline in SDXL?", "SDXL"),
    # 3D 生成
    ("How does Zero123 control the camera viewpoint for novel view synthesis?", "Zero123"),
    ("What dataset does Zero123 use for training?", "Zero123"),
    ("How does Zero123++ improve multi-view consistency over Zero123?", "Zero123++"),
    ("What is the multi-view consistency approach in Zero123++?", "Zero123++"),
    ("How does DreamFusion use score distillation sampling?", "DreamFusion"),
    ("What is the role of NeRF in DreamFusion?", "DreamFusion"),
    ("How does Magic3D achieve higher resolution than DreamFusion?", "Magic3D"),
    ("What two-stage approach does Magic3D use?", "Magic3D"),
    ("How does Point-E generate 3D point clouds from text?", "Point-E"),
    ("What is the clip-guided diffusion model in Point-E?", "Point-E"),
    ("How does Score Jacobian Chaining lift 2D diffusion priors to 3D?", "SJC"),
    ("What is the perturb-and-average scoring technique in SJC?", "SJC"),
    # NeRF 系列
    ("How does NeRF represent 3D scenes as neural radiance fields?", "NeRF"),
    ("What is the volume rendering equation used in NeRF?", "NeRF"),
    ("How does Instant-NGP use multiresolution hash encoding to speed up training?", "Instant-NGP"),
    ("What speedup does Instant-NGP achieve compared to original NeRF?", "Instant-NGP"),
    ("How does 3D Gaussian Splatting represent and render radiance fields?", "3DGS"),
    ("What is the tile-based rasterizer used in 3D Gaussian Splatting?", "3DGS"),
    # 多视角生成
    ("How does One-2-3-45 reconstruct a 3D mesh from a single image?", "One-2-3-45"),
    ("What is the role of Zero123 in the One-2-3-45 pipeline?", "One-2-3-45"),
    ("How does SyncDreamer ensure consistency across multiple generated views?", "SyncDreamer"),
    ("What is the synchronized noise prediction mechanism in SyncDreamer?", "SyncDreamer"),
    ("How does Wonder3D use cross-domain diffusion for single-image 3D generation?", "Wonder3D"),
    ("What is the multi-task diffusion model in Wonder3D?", "Wonder3D"),
]

NICKNAME_TO_TITLE = {
    "DDPM": "Denoising Diffusion Probabilistic Models",
    "DDIM": "Denoising Diffusion Implicit Models",
    "CFG": "Classifier-Free Diffusion Guidance",
    "Stable Diffusion": "High-Resolution Image Synthesis with Latent Diffusion Models",
    "ControlNet": "Adding Conditional Control to Text-to-Image Diffusion Models",
    "InstructPix2Pix": "InstructPix2Pix: Learning to Follow Image Editing Instructions",
    "Imagen": "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding",
    "SDXL": "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis",
    "Zero123": "Zero-1-to-3: Zero-shot One Image to 3D Object",
    "Zero123++": "Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model",
    "DreamFusion": "DreamFusion: Text-to-3D using 2D Diffusion",
    "Magic3D": "Magic3D: High-Resolution Text-to-3D Content Creation",
    "Point-E": "Point-E: A System for Generating 3D Point Clouds from Complex Prompts",
    "SJC": "Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",
    "NeRF": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
    "Instant-NGP": "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding",
    "3DGS": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
    "One-2-3-45": "One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization",
    "SyncDreamer": "SyncDreamer: Generating Multiview-consistent Images from a Single-view Image",
    "Wonder3D": "Wonder3D: Single Image to 3D using Cross-Domain Diffusion",
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
