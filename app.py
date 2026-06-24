import gradio as gr
from src.pipeline import ask

CORPUS = {
    "Diffusion Model Fundamentals": ["DDPM", "DDIM", "CFG", "Stable Diffusion"],
    "Diffusion Model Applications": ["ControlNet", "InstructPix2Pix", "Imagen", "SDXL"],
    "3D Generation": ["Zero123", "Zero123++", "DreamFusion", "Magic3D", "Point-E", "SJC"],
    "NeRF Series": ["NeRF", "Instant-NGP", "3D Gaussian Splatting"],
    "Multi-view Generation": ["One-2-3-45", "SyncDreamer", "Wonder3D"],
}

EXAMPLES = [
    "How does DDIM improve upon DDPM in terms of sampling speed?",
    "What is the role of the VAE in Stable Diffusion and why is it used?",
    "How does ControlNet add spatial conditioning without retraining the base model?",
    "How does score distillation sampling work in DreamFusion?",
    "What are the key differences between 3D Gaussian Splatting and NeRF?",
]


def format_corpus_md() -> str:
    lines = ["### Paper Corpus\n"]
    for category, papers in CORPUS.items():
        lines.append(f"**{category}**")
        for paper in papers:
            lines.append(f"- {paper}")
        lines.append("")
    return "\n".join(lines)


def respond(question, history):
    if not question.strip():
        return (
            history, "",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            "", "", "",
        )

    answer, sources, follow_ups = ask(question)

    unique_sources = list({s["title"] for s in sources})
    source_text = "\n".join([f"- {title}" for title in unique_sources])
    full_answer = f"{answer}\n\n**Sources:**\n{source_text}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_answer},
    ]

    fu = (follow_ups + ["", "", ""])[:3]

    return (
        history,
        "",
        gr.update(value=fu[0], visible=bool(fu[0])),
        gr.update(value=fu[1], visible=bool(fu[1])),
        gr.update(value=fu[2], visible=bool(fu[2])),
        fu[0], fu[1], fu[2],
    )


with gr.Blocks(title="ML Paper Assistant") as demo:
    gr.Markdown("# ML Paper Assistant")
    gr.Markdown(
        "RAG-powered Q&A over 20 ML papers. "
        "Ask anything about diffusion models, NeRF, or 3D generation."
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=180):
            gr.Markdown(format_corpus_md())

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, show_label=False)

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask a question about any paper...",
                    show_label=False,
                    scale=5,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            with gr.Row():
                followup_btns = [gr.Button(visible=False, size="sm") for _ in range(3)]

            followup_states = [gr.State("") for _ in range(3)]

            gr.Examples(examples=EXAMPLES, inputs=question_box)

    all_outputs = [chatbot, question_box] + followup_btns + followup_states

    submit_btn.click(respond, [question_box, chatbot], all_outputs)
    question_box.submit(respond, [question_box, chatbot], all_outputs)

    for btn, state in zip(followup_btns, followup_states):
        btn.click(respond, [state, chatbot], all_outputs)


if __name__ == "__main__":
    demo.launch()
