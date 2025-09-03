import streamlit as st
from src.rag_agent import RAGAgent
from src.config import TARGET_QUESTION


import sys



st.set_page_config(page_title="RAG Demo - Chinchilla", layout="centered")


st.title("📑 RAG Demo - Chinchilla chinchilla")


st.markdown(
    "Esta demo permite realizar consultas a la base documental vectorizada.\n\n"
    "Pregunta de ejemplo: **¿En qué proyectos fue relevante la chinchilla chinchilla?**"
)


query = st.text_area("✍️ Ingresa tu consulta:", value=TARGET_QUESTION, height=100)


if st.button("🔎 Consultar"):
    with st.spinner("Buscando respuesta..."):
        rag = RAGAgent(provider="groq") 
        response = rag.query(query)

    st.subheader("📌 Respuesta estructurada")
    st.json(response)
