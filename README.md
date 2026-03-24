# 📄 FastAPI Docs RAG Chatbot

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries based on FastAPI documentation. Instead of relying on general LLM knowledge, the system retrieves relevant context from a document corpus and generates grounded responses.

## ▶️ Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Ingest documents
python ingest.py

# Step 2: Run chatbot UI
streamlit run app.py
```

## 🧠 System Architecture

User Query  
→ Query Embedding  
→ FAISS Similarity Search (Top-K Retrieval)  
→ Context Aggregation  
→ LLM Response Generation

## 🛠 Tech Stack

- Python
- LangChain (modular ecosystem)
- OpenAI API (embeddings + generation)
- FAISS (vector database)
- Streamlit (chat UI)

## ⚙️ Pipeline Breakdown

### 1. Document Ingestion

- Load FastAPI documentation (PDF)
- Split into overlapping chunks
- Generate embeddings using text-embedding-3-small
- Store vectors in FAISS

### 2. Retrieval

- Convert user query into embedding
- Perform similarity search over FAISS
- Retrieve top-k relevant chunks

### 3. Generation

- Construct context from retrieved chunks
- Pass context + query to LLM (gpt-5.4-nano)
- Enforce grounded responses via prompt constraints

## 💬 Example Queries

- What is FastAPI?
- How to create a route in FastAPI?
- How to run a FastAPI application?
- What is dependency injection in FastAPI?

## ⚠️ Challenges & Observations

- Retrieval mismatch: user queries often differ from documentation phrasing
- Improved by increasing top-k retrieval and refining query structure
- Enforced strict grounding to prevent hallucinations
- Trade-off between recall (higher k) and noise in context

## 🔮 Future Improvements

- Add re-ranking for better retrieval accuracy
- Introduce hybrid search (keyword + embedding)
- Wrap with FastAPI for production backend