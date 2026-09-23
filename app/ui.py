"""Streamlit entry point for RepoGuide."""

import streamlit as st


st.set_page_config(page_title="RepoGuide", page_icon="🔎")

st.title("RepoGuide")
st.caption("Local RAG assistant for understanding Python repositories.")

repo_path = st.text_input("Repository path")

if repo_path:
    st.info("Indexing pipeline not connected yet. Start with app.ingest and app.chunker.")

question = st.text_input("Ask a question about the repository")

if question:
    st.warning("Generation is not connected yet. Implement retrieval and Llama inference first.")
