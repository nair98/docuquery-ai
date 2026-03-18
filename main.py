# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from ai.engine import CloudAI, OllamaAI
import os

app = FastAPI(title="DocuQuery AI")

# Initialize the pipeline once
pipeline = RAGPipeline()

# Initialize AI backend based on env variable
backend = os.getenv("AI_BACKEND", "CLOUD").upper()

if backend == "OLLAMA":
    ai = OllamaAI()
else:
    ai = CloudAI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_docs(request: QueryRequest):
    # 1. Get search results from RAG pipeline (optional context)
    search_results = pipeline.search(request.query)

    # 2. You can format or combine search_results as needed for AI input
    # For now, let's keep it simple: just pass the original query to AI
    ai_response = ai.generate_text(request.query)

    # 3. Return only the AI response as clean JSON
    return {"response": ai_response}

@app.get("/")
def root():
    return {"message": "Welcome to DocuQuery AI! Use /query to search your documents."}