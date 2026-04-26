# PapersRag

A local RAG (Retrieval-Augmented Generation) pipeline for querying AI research papers using LangChain, FAISS, and local LLM.

## Dataset (download từ arxiv)

| Paper | Domain |
|---|---|
| A Comprehensive Survey on Vector Data... | Vector DB / Search |
| An Introduction to Vision-Language Models | VLM |
| Attention Is All You Need | Transformer |
| DeepSeek-VL Towards Real-World Vision... | VLM |
| LangChain | RAG Framework |
| LLaMA Open and Efficient Foundation Models | LLM |
| PoseNet A Convolutional Network for Re... | Computer Vision |
| Qwen3 Technical Report | LLM |
| Searching for MobileNetV3 | Efficient CNN |
| WHAT IS YOLOV8 | Object Detection |
| YOLO26 | Object Detection |

---

## Changelog

### v0.0.1 — Initial setup with Gemini Embedding + Chat
- Stack ban đầu: Google Gemini Embedding + `gemini-2.0-flash` LLM
- **Lỗi:** Free tier embedding giới hạn 100 req/min → thêm `BATCH_SIZE=30` + `SLEEP_SEC=65s` để tránh rate limit

### v0.0.2 — Embedding quota exhausted
- 11 PDFs => 270 pages => 1060 chunks => hết 1000 req/day
- **Fix:** Chuyển sang HuggingFace local embedding `sentence-transformers/all-MiniLM-L6-v2` (ko rate limit, offline)
  - Không cần API key, không rate limit, chạy offline
  - Xóa`SLEEP_SEC`trong `vectorstore.py`
  - Tăng `BATCH_SIZE` từ 30 → 100

### v0.0.3 — LLM quota exhausted
- First query → 429 RESOURCE_EXHAUSTED trên `gemini-2.0-flash` free tier
- **Fix:** Chuyển LLM sang Ollama local `llama3.2` (ko quota, offline)
  - Cài Ollama app → `ollama pull llama3.2` → `pip install langchain-ollama`
  - Time response khá lâu 
---

## How to test
```bash
git clone <repo>
cd PapersRag
python -m venv myenv
myenv\Scripts\Activate.ps1
pip install -r requirements.txt

# Chuyển các Papers(hoặc Docs) dạng file (.pdf) vào trong Dataset/

# Cài Ollama
irm https://ollama.com/install.ps1 | iex
ollama pull llama3.2

python main.py
```