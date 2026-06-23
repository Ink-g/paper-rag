import gradio as gr
from src.pipeline import ask


def chat(question: str, history: list) -> str:
    if not question.strip():
        return "Please enter a question."

    answer, sources = ask(question)

    unique_sources = list({s["title"] for s in sources})
    source_text = "\n".join([f"- {title}" for title in unique_sources])

    return f"{answer}\n\n**Sources:**\n{source_text}"


demo = gr.ChatInterface(
    fn=chat,
    title="ML Paper Assistant",
    description="RAG-powered Q&A over 20 ML papers (diffusion models, NeRF, 3D generation, etc.) with vector retrieval + reranker",
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
