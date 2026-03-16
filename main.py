from fastapi import FastAPI, UploadFile
from rag_pipeline import process_document, ask_question

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile):
    content = await file.read()
    process_document(content)
    return {"message": "Document processed"}

@app.post("/ask")
async def ask(query: str):
    answer = ask_question(query)
    return {"answer": answer}