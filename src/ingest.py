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
    # 参考文献一般在最后，找到 "References" 标题后截断
    match = re.search(r'\n(References|REFERENCES|Bibliography)\n', text)
    if match:
        return text[:match.start()]
    return text


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    # 按段落切分，段落之间有空行
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果加上这个段落还没超过 chunk_size，就继续拼
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # 新 chunk 从上一个 chunk 的尾部开始（overlap），保留上下文
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
            print(f"找不到文件，跳过: {pdf_path}")
            continue

        print(f"处理: {paper['nickname']} - {paper['title']}")

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

        print(f"  切成 {len(chunks)} 个文本块")

    print(f"\n共处理 {len(metadata)} 篇论文，生成 {len(all_chunks)} 个文本块")
    return all_chunks


def store_chunks(chunks: list[Chunk]):
    print("\n加载 embedding 模型（首次运行会下载约 2GB，请耐心等待）...")
    embed_model = SentenceTransformer("BAAI/bge-m3")

    chroma_client = chromadb.PersistentClient(path="./storage")
    # 每次重新入库前清空旧数据，避免重复
    chroma_client.delete_collection("papers")
    collection = chroma_client.get_or_create_collection("papers")

    print("正在生成向量并存入数据库...")
    texts = [c.text for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=8)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"paper_id": c.paper_id, "title": c.title, "section": c.section} for c in chunks],
        ids=[c.id for c in chunks]
    )
    print(f"完成，共存入 {len(chunks)} 个文本块。")


if __name__ == "__main__":
    chunks = ingest_all()
    store_chunks(chunks)
