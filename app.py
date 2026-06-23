import gradio as gr
from src.pipeline import ask


def chat(question: str, history: list) -> str:
    if not question.strip():
        return "请输入问题。"

    answer, sources = ask(question)

    unique_sources = list({s["title"] for s in sources})
    source_text = "\n".join([f"- {title}" for title in unique_sources])

    return f"{answer}\n\n**来源论文：**\n{source_text}"


demo = gr.ChatInterface(
    fn=chat,
    title="ML 论文助手",
    description="基于 20 篇 ML 论文（扩散模型、NeRF、3D 生成等）的问答系统，支持向量检索 + Reranker 精排",
    examples=[
        "What is the main idea of DDPM and how does the denoising process work?",
        "How does ControlNet add spatial conditioning to diffusion models?",
        "What are the advantages of 3D Gaussian Splatting over NeRF?",
        "What is classifier-free guidance and why is it useful?",
        "How does DreamFusion generate 3D objects from text?",
    ],
)

if __name__ == "__main__":
    demo.launch()
