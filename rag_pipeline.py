# rag_pipeline.py
import os
from sentence_transformers import SentenceTransformer
import pickle

class RAGPipeline:
    def __init__(self, data_folder="data", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_folder = data_folder
        self.model = SentenceTransformer(model_name)
        self.docs = []
        self.embeddings = None
        self.load_documents()
        self.create_embeddings()

    def load_documents(self):
        for file in os.listdir(self.data_folder):
            path = os.path.join(self.data_folder, file)
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    self.docs.append(f.read())
            elif file.endswith(".pdf"):
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    self.docs.append(text)
                except:
                    print(f"Failed to read {file}")
            elif file.endswith(".docx"):
                try:
                    import docx
                    doc = docx.Document(path)
                    text = "\n".join([p.text for p in doc.paragraphs])
                    self.docs.append(text)
                except:
                    print(f"Failed to read {file}")

    def create_embeddings(self):
        if self.docs:
            self.embeddings = self.model.encode(self.docs, convert_to_tensor=True)
            # Optional: save embeddings for later
            with open("embeddings.pkl", "wb") as f:
                pickle.dump(self.embeddings, f)
        else:
            print("No documents found to embed.")

    def search(self, query, top_k=3):
        import torch
        query_emb = self.model.encode([query], convert_to_tensor=True)
        if self.embeddings is None:
            print("Embeddings not found.")
            return []
        # Compute cosine similarity
        cos_scores = torch.nn.functional.cosine_similarity(query_emb, self.embeddings)
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.docs)))
        results = [self.docs[i] for i in top_results.indices]
        return results