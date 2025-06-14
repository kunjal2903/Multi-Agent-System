
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from app.pdf_utils import load_and_split_pdf
from app.vector_store import create_vector_store, save_vector_store, load_vector_store
from app.query_engine import query_pdf
import os

app = FastAPI()
VECTOR_STORE_PATH = "faiss_index"

@app.get("/")
async def print_command():
    return {"message": "Hello from the root endpoint"}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = load_and_split_pdf(file_path)
    vector_store = create_vector_store(chunks)
    save_vector_store(vector_store, VECTOR_STORE_PATH)

    os.remove(file_path)
    return {"message": "PDF processed and indexed successfully."}

@app.post("/query/")
async def query_document(query: str = Form(...)):
    try:
        vector_store = load_vector_store(VECTOR_STORE_PATH)
        answer = query_pdf(vector_store, query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
