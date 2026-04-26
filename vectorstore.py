import os
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

USE_FAISS = True

FAISS_PATH = "./vectorstore/faiss_index"
CHROMA_PATH = "./vectorstore/chroma_index"
BATCH_SIZE = 100

os.makedirs("./vectorstore", exist_ok=True)

def build_or_load(splits, embeddings):
    if USE_FAISS:
        if os.path.exists(FAISS_PATH) and os.listdir(FAISS_PATH):
            print("Loading FAISS from disk...")
            return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

        total_batches = -(-len(splits) // BATCH_SIZE)
        print(f"Building FAISS... ({len(splits)} chunks, {total_batches} batches)")

        db = FAISS.from_documents(splits[:BATCH_SIZE], embeddings)
        print(f"[1/{total_batches}] done")

        for i in range(BATCH_SIZE, len(splits), BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch = splits[i:i + BATCH_SIZE]
            db.add_documents(batch)
            print(f"[{batch_num}/{total_batches}] done")

        db.save_local(FAISS_PATH)
        print("FAISS saved to disk.")
        return db

    else:
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
            print("Loading Chroma from disk...")
            return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("Building Chroma...")
        return Chroma.from_documents(splits, embeddings, persist_directory=CHROMA_PATH)