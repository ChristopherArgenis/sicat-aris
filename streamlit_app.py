import streamlit as st
import google.genai as genai
from chromadb import PersistentClient
import os

# Load Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-1.5-flash"

# Load Chroma
client = PersistentClient(path="chroma_db")
collection = client.get_collection("tickets_28k")

# UI
st.title("🔮 SICAT-ARIS — Demo Semántica")
query = st.text_input("Escribe tu consulta de ticket:")

if st.button("Buscar"):
    q_emb = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5
    )

    docs = results["documents"][0]

    st.subheader("Resultados similares:")
    for d in docs:
        st.write("- ", d)

    # Chain-of-thought (controlado)
    llm_response = genai.generate_text(
        model=MODEL,
        prompt=f"""
            Identifica el problema del ticket basado en:
            {docs}

            Responde breve y técnico.
        """
    )
    st.subheader("Interpretación ARIS:")
    st.write(llm_response.text)
