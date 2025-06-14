from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# You can choose any model from sentence-transformers (e.g., all-MiniLM-L6-v2)

from langchain.embeddings import HuggingFaceEmbeddings

def get_hf_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")


def create_vector_store(chunks):
    embeddings = get_hf_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def save_vector_store(vector_store, path="faiss_index"):
    vector_store.save_local(path)

def load_vector_store(path="faiss_index"):
    embeddings = get_hf_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


