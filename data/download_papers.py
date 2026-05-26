import arxiv
import json
import os
import urllib.request

papers = [
    # 扩散模型基础
    ("2006.11239", "DDPM"),
    ("2010.02502", "DDIM"),
    ("2207.12598", "CFG"),
    ("2112.10752", "Stable Diffusion"),
    # 3D 生成
    ("2303.11328", "Zero123"),
    ("2310.15110", "Zero123++"),
    ("2209.14988", "DreamFusion"),
    ("2211.10440", "Magic3D"),
    ("2212.08751", "Point-E"),
]

os.makedirs("data/papers", exist_ok=True)
metadata = []

client = arxiv.Client()

for paper_id, nickname in papers:
    print(f"下载中: {nickname} ({paper_id})")
    result = next(client.results(arxiv.Search(id_list=[paper_id])))
    urllib.request.urlretrieve(result.pdf_url, f"data/papers/{nickname}.pdf")
    metadata.append({
        "paper_id": paper_id,
        "nickname": nickname,
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "year": result.published.year,
        "abstract": result.summary,
        "filename": f"{nickname}.pdf"
    })
    print(f"完成: {result.title}")

with open("data/papers/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n全部完成，共下载 {len(metadata)} 篇论文。")
