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
        "How does DDIM improve upon DDPM in terms of sampling speed?",
        "What is the role of the VAE in Stable Diffusion and why is it used?",
        "How does ControlNet add spatial conditioning without retraining the base model?",
        "How does score distillation sampling work in DreamFusion?",
        "What are the key differences between 3D Gaussian Splatting and NeRF?",
    ],
)

if __name__ == "__main__":
    demo.launch()
