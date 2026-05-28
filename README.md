# ML 论文助手

基于 RAG（检索增强生成）技术的机器学习论文问答系统。输入问题，系统自动从论文库中检索相关内容，由本地大语言模型生成回答并标注来源论文。

## 论文库

| 方向 | 论文 |
|------|------|
| 扩散模型 | DDPM、DDIM、CFG、Stable Diffusion |
| 3D 生成 | Zero123、Zero123++、DreamFusion、Magic3D、Point-E |

## 技术架构

```
用户提问
   ↓
bge-m3 将问题转为向量
   ↓
ChromaDB 检索最相关的 5 个文本块
   ↓
Qwen2.5-7B（本地）生成回答
   ↓
返回答案 + 来源论文
```

## 环境要求

- Python 3.11
- Anaconda / Miniconda
- Ollama（本地 LLM）
- NVIDIA GPU（推荐，CPU 也可运行但较慢）

---

## 快速开始

### 第一次使用

**1. 激活虚拟环境**

```powershell
conda activate paper-rag
```

**2. 启动 Ollama**

Ollama 安装后会自动在后台运行。如果没有运行，打开 Ollama 应用即可。
确认 Qwen 模型已下载：

```powershell
ollama pull qwen2.5:7b
```

**3. 启动界面**

```powershell
cd D:\Projects\paper-rag
python app.py
```

浏览器打开 `http://127.0.0.1:7860` 即可使用。

---

### 重新构建向量数据库

如果 `storage/` 目录被删除，或新增了论文，需要重新入库：

```powershell
conda activate paper-rag
cd D:\Projects\paper-rag
python src/ingest.py
```

首次运行会自动下载 bge-m3 模型（约 2GB），之后会缓存在本地。

---

### 下载新论文

编辑 `data/download_papers.py`，在列表里添加新的 arxiv ID：

```python
("2303.11328", "Zero123"),   # arxiv ID, 自定义名称
```

然后运行：

```powershell
python data/download_papers.py
python src/ingest.py          # 重新入库
```

---

## 项目结构

```
paper-rag/
├── data/
│   ├── papers/              # PDF 文件和 metadata.json
│   └── download_papers.py   # 论文下载脚本
├── storage/                 # ChromaDB 向量数据库（本地，不进 git）
├── src/
│   ├── ingest.py            # PDF 解析、切块、向量入库
│   ├── retriever.py         # 向量检索
│   ├── pipeline.py          # RAG 主流程（检索 + 生成）
│   └── evaluate.py          # 评估（待实现）
├── app.py                   # Gradio 前端
├── test_pipeline.py         # 命令行测试
├── requirements.txt         # Python 依赖
└── .env                     # API Key（不进 git）
```

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 激活环境 | `conda activate paper-rag` |
| 启动界面 | `python app.py` |
| 命令行提问 | `python test_pipeline.py` |
| 重新入库 | `python src/ingest.py` |
| 下载论文 | `python data/download_papers.py` |
| 退出环境 | `conda deactivate` |
