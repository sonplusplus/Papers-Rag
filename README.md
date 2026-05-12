# PapersRag

A local RAG (Retrieval-Augmented Generation) pipeline for querying AI research papers using LangChain, FAISS, and local LLM.

## Architecture

```
PDF Files
    ↓  DirectoryLoader + PyPDFLoader
Raw Documents
    ↓  RecursiveCharacterTextSplitter (chunk_size=1200, overlap=300)
Chunks (~1060 total from 11 papers)
    ↓
    ├── BM25 Index (sparse, keyword-exact)
    └── FAISS Index (dense, semantic)
            ↓  EnsembleRetriever (RRF weights: BM25=0.4, FAISS=0.6)
Top-10 Candidates
    ↓  CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)
Top-3 Reranked Chunks
    ↓  PromptTemplate + ChatOllama (llama3.2)
Grounded Answer with Citations
```



## How to run

### 1. Clone & install

```bash
git clone https://github.com/sonplusplus/Papers-Rag
python -m venv myenv

# Windows
myenv\Scripts\Activate.ps1

# Mac/Linux
source myenv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` includes `rank-bm25` (needed for BM25Retriever) and `sentence-transformers` (needed for CrossEncoder reranker).

---

### 2. Add papers

Copy PDF files into the `Dataset/` folder:

```
PapersRag/
└── Dataset/
    ├── Attention Is All You Need.pdf
    ├── LLaMA.pdf
    └── ...
```

---

### 3. Install and start Ollama

```bash

irm https://ollama.com/install.ps1 | iex


brew install ollama

ollama pull llama3.2

ollama serve
```

---

### 4. Run the QA interface

```bash
python main.py
```

First run: loads PDFs → chunks → builds FAISS index → saves to `vectorstore/faiss_index/`  
Subsequent runs: loads FAISS from disk (fast).

Expected output:
```
Loaded 11 docs → 1060 chunks
Loading FAISS from disk...
Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
Retriever: Hybrid BM25+FAISS + cross-encoder reranker (top 3)

Question (q to quit): What is multi-head attention?
Answer: Multi-head attention allows the model to ... (source: Attention Is All You Need.pdf | page: 4)
```

---

### 5. Run retrieval evaluation

```bash
python eval.py
```


---

## Configuration

In `main.py`:
```python
USE_RERANKER = True   
                      
```

In `retriever.py`:
```python
build_hybrid_retriever(..., bm25_weight=0.4, faiss_weight=0.6, k=10)
```