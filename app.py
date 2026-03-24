import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# PAGE CONFIG
st.set_page_config(page_title="Construction RAG Assistant", layout="wide")

st.title("Construction AI Assistant")

st.write("Ask questions about construction policies, delays, quality, payments, etc.")

# LOAD MODELS
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_store/faiss.index")

    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    llm = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=150
    )

    return embed_model, index, chunks, llm

embed_model, index, chunks, llm = load_models()

# RETRIEVAL
def retrieve(query, k=2):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

# RAG ANSWER
def generate_answer(query):

    ctx = retrieve(query)
    context_text = "\n\n".join(ctx)

    prompt = f"""
You are a construction assistant.

Rules:
- Answer ONLY using provided context
- If answer not found say: Not found in documents
- Be concise and professional

Context:
{context_text}

Question:
{query}

Final Answer:
"""

    output = llm(prompt)[0]["generated_text"]
    answer = output.split("Final Answer:")[-1].strip()

    return ctx, answer

# CHAT UI
query = st.text_input("Ask your question")

if st.button("Get Answer"):

    with st.spinner("Searching documents..."):
        ctx, answer = generate_answer(query)

    st.subheader("Retrieved Context")
    for c in ctx:
        st.write(c[:500] + "...")

    st.subheader("Final Answer")
    st.success(answer)