from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder


def build_hybrid_retriever(splits, faiss_db, bm25_weight=0.4, faiss_weight=0.6, k=10):
    """
    Returns an EnsembleRetriever that merges BM25 and FAISS rankings
    via Reciprocal Rank Fusion.

    Weights:
      bm25_weight=0.4  — keyword precision (paper names, acronyms, exact terms)
      faiss_weight=0.6 — semantic recall (paraphrases, concept-level queries)

    k=10: retrieve more candidates for the reranker to work with.
    """
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = k

    faiss_retriever = faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[bm25_weight, faiss_weight]
    )
    return ensemble

class CrossEncoderReranker:


    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_n: int = 3) -> list:
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]


class HybridRerankedRetriever:
    """
    Full two-stage retriever.
    Call .invoke(query) to get top_n reranked documents.
    """
    def __init__(self, splits, faiss_db, top_n=3):
        self.hybrid = build_hybrid_retriever(splits, faiss_db, k=10)
        self.reranker = CrossEncoderReranker()
        self.top_n = top_n

    def invoke(self, query: str) -> list:
        candidates = self.hybrid.invoke(query)
        return self.reranker.rerank(query, candidates, top_n=self.top_n)
