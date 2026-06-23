# ML Paper Assistant

A RAG (Retrieval-Augmented Generation) system for machine learning research papers. Ask a question and the system retrieves relevant content from the paper corpus, generates an answer using a local LLM, and cites the source papers.

## Paper Corpus

| Category | Papers |
|----------|--------|
| Diffusion Model Fundamentals | DDPM, DDIM, CFG, Stable Diffusion |
| Diffusion Model Applications | ControlNet, InstructPix2Pix, Imagen, SDXL |
| 3D Generation | Zero123, Zero123++, DreamFusion, Magic3D, Point-E, SJC |
| NeRF Series | NeRF, Instant-NGP, 3D Gaussian Splatting |
| Multi-view Generation | One-2-3-45, SyncDreamer, Wonder3D |

## Architecture

```
User question
   ↓
bge-m3 encodes question into vector
   ↓
ChromaDB vector search, retrieve 20 candidates
   ↓
bge-reranker-v2-m3 reranks, select Top 5
   ↓
Qwen2.5-7B (local) generates answer
   ↓
Return answer + source papers
```

## Requirements

- Python 3.11
- Anaconda / Miniconda
- Ollama (local LLM)
- NVIDIA GPU (recommended; CPU also works but slower)

---

## Quick Start

### First-time Setup

**1. Activate the environment**

```powershell
conda activate paper-rag
```

**2. Start Ollama**

Ollama runs in the background after installation. If it's not running, open the Ollama app.
Confirm the Qwen model is downloaded:

```powershell
ollama pull qwen2.5:7b
```

**3. Launch the interface**

```powershell
cd D:\Projects\paper-rag
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

### Rebuild the Vector Database

If the `storage/` directory is deleted or new papers are added:

```powershell
conda activate paper-rag
cd D:\Projects\paper-rag
python src/ingest.py
```

The bge-m3 model (~2GB) is downloaded on first run and cached locally.

---

### Download New Papers

Edit `data/download_papers.py` and add a new arxiv ID to the list:

```python
("2303.11328", "Zero123"),   # arxiv ID, custom name
```

Then run:

```powershell
python data/download_papers.py
python src/ingest.py
```

---

## Project Structure

```
paper-rag/
├── data/
│   ├── papers/              # PDF files and metadata.json
│   └── download_papers.py   # Paper download script
├── storage/                 # ChromaDB vector database (local, not in git)
├── src/
│   ├── ingest.py            # PDF parsing, chunking, embedding, storage
│   ├── retriever.py         # Vector retrieval + reranking
│   ├── pipeline.py          # RAG pipeline (retrieval + generation)
│   └── evaluate.py          # Evaluation script (Hit Rate@5)
├── app.py                   # Gradio frontend
├── test_pipeline.py         # CLI testing
├── requirements.txt         # Python dependencies
└── .env                     # API keys (not in git)
```

## Evaluation Results

Hit Rate@5 evaluated on 42 manually annotated questions (correct paper appears in top 5 retrieved chunks):

| Method | Hit Rate@5 |
|--------|-----------|
| Vector retrieval (baseline) | 85.7% |
| Vector retrieval + Reranker | 90.5% |

> On a 20-paper, 217-chunk corpus, the reranker improves over the baseline by 4.8 percentage points. The advantage of reranking becomes more pronounced as the corpus grows.

## Common Commands

| Action | Command |
|--------|---------|
| Activate environment | `conda activate paper-rag` |
| Launch UI | `python app.py` |
| CLI query | `python test_pipeline.py` |
| Re-ingest papers | `python src/ingest.py` |
| Download papers | `python data/download_papers.py` |
| Deactivate environment | `conda deactivate` |
