import streamlit as st

def get_app_notes():
    return st.markdown(
        """
        - This is a lightweight product which was designed to try VLM embeddings \
        in a RAG framework to retrieve insights on mathematically / graphically \
        heavy papers, as is common in current ML/RL papers. 
        """
    )