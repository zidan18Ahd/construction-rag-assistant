# Construction RAG Assistant

This project implements a **Retrieval-Augmented Generation (RAG) based AI assistant** for a construction marketplace.  
The assistant answers user queries strictly using internal policy and specification documents.

Deployed App:  
https://construction-rag-assistant-6poxbjyjro4vwbvyfgmvmi.streamlit.app/

---

## Objective

The goal of this project is to build a simple RAG pipeline that:

- Retrieves relevant information from internal documents
- Generates grounded answers using retrieved context
- Ensures transparency and explainability in responses

---

##  Input Documents

The system uses internal company documents including:

- Customer protection policies
- Quality assurance systems
- Construction workflow and project management guidelines

---

##  System Architecture

User Query → Embedding → Vector Search → Context Retrieval → LLM → Grounded Answer

---

## Embedding Model

**sentence-transformers/all-MiniLM-L6-v2**

Reason for selection:

- Lightweight and fast
- Good semantic similarity performance
- Suitable for local deployment

---

## Document Chunking

Documents are split into overlapping chunks:

- Chunk size: ~120 words
- Overlap: ~30 words

This improves retrieval granularity and relevance.

---

## Vector Search

Vector indexing implemented using:

**FAISS (Inner Product Similarity)**

Retrieval process:

- Query embedding generated
- Top-k most relevant chunks retrieved
- Retrieved chunks passed to LLM

---

## LLM Used

**TinyLlama (Local Open-Source Model)**

Reason:

- Enables fully local inference
- Demonstrates open-source RAG capability
- Avoids reliance on external APIs

Grounding enforcement:

The LLM is prompted to:

- Answer only using retrieved context
- Avoid hallucinated or unsupported claims

---

## Transparency & Explainability

The system explicitly displays:

- Retrieved document chunks
- Final generated grounded answer

This ensures traceability of responses.

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
