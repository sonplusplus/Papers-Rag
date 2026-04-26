import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from load_paper import load_and_chunk
from vectorstore import build_or_load

load_dotenv()

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source','?')} | page {doc.metadata.get('page','?')}]\n{doc.page_content}"
        for doc in docs
    )

def main():
    splits = load_and_chunk()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = build_or_load(splits, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOllama(model="llama3.2", temperature=0)
    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    while True:
        query = input("\nQuestion (q to quit): ").strip()
        if query.lower() == "q":
            break
        print("\nAnswer:", rag_chain.invoke(query))

if __name__ == "__main__":
    main()