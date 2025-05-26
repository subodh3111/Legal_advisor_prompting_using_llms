#  Legal Advisor AI Chatbot

This is a full-stack AI chatbot that provides legal guidance using your custom legal documents. It leverages **LangChain** for Retrieval-Augmented Generation (RAG), **FAISS** for vector storage, and **distilgpt2** (a Hugging Face model) for answer generation. The frontend is built using **React** and the backend uses **FastAPI**.

---

##  Features

-  Ask legal questions and get context-aware answers
-  Uses your content (`content.txt`) for retrieval
-  Lightweight and runs locally (no API keys required)
-  Vector similarity search with FAISS
-  HuggingFace model (`distilgpt2`) for generating answers

---

##  Project Structure


---

## ⚙️ Backend Setup (FastAPI)

1. **Create a virtual environment (optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate


