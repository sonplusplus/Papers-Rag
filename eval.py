import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from load_paper import load_and_chunk
from vectorstore import build_or_load
from retriever import build_hybrid_retriever

load_dotenv()


QA_PAIRS = [
    {
        "question": "What positional encoding method does the original Transformer use?",
        "expected_source_keywords": ["Attention"],
        "note": "Attention Is All You Need"
    },
    {
        "question": "What is the multi-head attention mechanism?",
        "expected_source_keywords": ["Attention"],
        "note": "Attention Is All You Need"
    },
    {
        "question": "How does LLaMA handle tokenization?",
        "expected_source_keywords": ["LLaMA"],
        "note": "LLaMA paper"
    },
    {
        "question": "What efficiency techniques does MobileNetV3 use?",
        "expected_source_keywords": ["MobileNet"],
        "note": "Searching for MobileNetV3"
    },
    {
        "question": "What loss function does YOLOv8 use for bounding box regression?",
        "expected_source_keywords": ["YOLO"],
        "note": "YOLO papers"
    },
    {
        "question": "What is Mixture of Experts and how does Qwen3 use it?",
        "expected_source_keywords": ["Qwen"],
        "note": "Qwen3 Technical Report"
    },
    {
        "question": "What are the differences between HNSW and IVF vector indexing?",
        "expected_source_keywords": ["Vector"],
        "note": "Vector DB survey"
    },
    {
        "question": "What task does PoseNet solve and what is its architecture?",
        "expected_source_keywords": ["PoseNet"],
        "note": "PoseNet paper"
    },
    {
        "question": "How does DeepSeek-VL handle interleaved image-text input?",
        "expected_source_keywords": ["DeepSeek"],
        "note": "DeepSeek-VL paper"
    },
    {
        "question": "What is retrieval-augmented generation and how does LangChain implement it?",
        "expected_source_keywords": ["LangChain"],
        "note": "LangChain paper"
    },
]

OUT_OF_SCOPE = [
    "What is the PERCLOS algorithm for drowsiness detection?",
    "How does ESP32-CAM perform INT8 inference?",
    "What is the current price of Bitcoin?",
]



def source_hit(docs, keywords: list[str]) -> bool:
    """Returns True if any expected keyword appears in any retrieved source path."""
    sources = " ".join(doc.metadata.get("source", "") for doc in docs)
    return any(kw.lower() in sources.lower() for kw in keywords)


def evaluate_retrieval(retriever, qa_pairs: list, k_label: str = "") -> float:
    print(f"RETRIEVAL EVALUATION {k_label}")
    print(f"{'='*10}")

    hits = 0
    for i, qa in enumerate(qa_pairs):
        docs = retriever.invoke(qa["question"])
        hit = source_hit(docs, qa["expected_source_keywords"])
        hits += int(hit)
        status = "✓ HIT " if hit else "✗ MISS"
        retrieved_sources = [
            os.path.basename(d.metadata.get("source", "?")) for d in docs[:3]
        ]
        print(f"[{i+1:02d}] {status} | {qa['note']}")
        print(f"       Q: {qa['question'][:70]}...")
        print(f"       Retrieved: {retrieved_sources}")

    recall = hits / len(qa_pairs)
    print(f"\nRetrieval Recall@k: {hits}/{len(qa_pairs)} = {recall:.1%}")
    return recall


def compare_retrievers(splits, faiss_db):
    """Compare dense-only vs hybrid retrieval side by side."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    dense_retriever = faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    hybrid_retriever = build_hybrid_retriever(splits, faiss_db, k=10)

    recall_dense = evaluate_retrieval(dense_retriever, QA_PAIRS, "— Dense Only (FAISS k=5)")
    recall_hybrid = evaluate_retrieval(hybrid_retriever, QA_PAIRS, "— Hybrid BM25+FAISS k=10")
    print(f"SUMMARY")
    print(f"  Dense-only recall:  {recall_dense:.1%}")
    print(f"  Hybrid recall:      {recall_hybrid:.1%}")
    delta = recall_hybrid - recall_dense
    print(f"  Improvement:        {delta:+.1%}")
    print(f"{'='*10}")

    return recall_dense, recall_hybrid


if __name__ == "__main__":
    print("Loading documents and building vector store...")
    splits = load_and_chunk()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_db = build_or_load(splits, embeddings)

    compare_retrievers(splits, faiss_db)
