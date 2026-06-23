import arxiv
import json
import os
import urllib.request

papers = [
    # Diffusion model fundamentals
    ("2006.11239", "DDPM"),
    ("2010.02502", "DDIM"),
    ("2207.12598", "CFG"),
    ("2112.10752", "Stable Diffusion"),
    # Diffusion model applications
    ("2302.05543", "ControlNet"),
    ("2211.09800", "InstructPix2Pix"),
    ("2205.11487", "Imagen"),
    ("2307.01952", "SDXL"),
    # 3D generation
    ("2303.11328", "Zero123"),
    ("2310.15110", "Zero123++"),
    ("2209.14988", "DreamFusion"),
    ("2211.10440", "Magic3D"),
    ("2212.08751", "Point-E"),
    ("2212.00774", "SJC"),
    # NeRF series
    ("2003.08934", "NeRF"),
    ("2201.05989", "Instant-NGP"),
    ("2308.04079", "3DGS"),
    # Multi-view generation
    ("2306.16928", "One-2-3-45"),
    ("2309.03453", "SyncDreamer"),
    ("2310.15008", "Wonder3D"),
]

os.makedirs("data/papers", exist_ok=True)

# Load existing metadata to skip already-downloaded papers
existing_metadata = []
metadata_path = "data/papers/metadata.json"
if os.path.exists(metadata_path):
    with open(metadata_path, encoding="utf-8") as f:
        existing_metadata = json.load(f)

existing_ids = {p["paper_id"] for p in existing_metadata}
metadata = existing_metadata.copy()

client = arxiv.Client()
new_count = 0

for paper_id, nickname in papers:
    if paper_id in existing_ids:
        print(f"Skipping (already exists): {nickname}")
        continue

    print(f"Downloading: {nickname} ({paper_id})")
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
    print(f"Done: {result.title}")
    new_count += 1

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\nDownloaded {new_count} new papers, {len(metadata)} total.")
