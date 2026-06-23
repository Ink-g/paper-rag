import fitz
import json
import re
import os
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb

PAPERS_DIR = "data/papers"
METADATA_PATH = "data/papers/metadata.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    id: str
    text: str
    paper_id: str
    title: str
    section: str


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


def remove_references(text: str) -> str:
    match = re.search(r'\n(References|REFERENCES|Bibliography)\n', text)
    if match:
        return text[:match.start()]
    return text


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + " " + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def ingest_all() -> list[Chunk]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    all_chunks = []

    for paper in metadata:
        pdf_path = os.path.join(PAPERS_DIR, paper["filename"])
        if not os.path.exists(pdf_path):
            print(f"File not found, skipping: {pdf_path}")
            continue

        print(f"Processing: {paper['nickname']} - {paper['title']}")

        text = extract_text(pdf_path)
        text = remove_references(text)
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                id=f"{paper['paper_id']}_{i}",
                text=chunk_text,
                paper_id=paper["paper_id"],
                title=paper["title"],
                section=f"chunk_{i}"
            )
            all_chunks.append(chunk)

        print(f"  -> {len(chunks)} chunks")

    print(f"\nProcessed {len(metadata)} papers, generated {len(all_chunks)} chunks total")
    return all_chunks


def store_chunks(chunks: list[Chunk]):
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer("BAAI/bge-m3")

    chroma_client = chromadb.PersistentClient(path="./storage")
    # Clear existing data to avoid duplicates on re-ingest
    chroma_client.delete_collection("papers")
    collection = chroma_client.get_or_create_collection("papers")

    print("Generating embeddings and storing to database...")
    texts = [c.text for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=8)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"paper_id": c.paper_id, "title": c.title, "section": c.section} for c in chunks],
        ids=[c.id for c in chunks]
    )
    print(f"Done. Stored {len(chunks)} chunks.")


if __name__ == "__main__":
    chunks = ingest_all()
    store_chunks(chunks)
