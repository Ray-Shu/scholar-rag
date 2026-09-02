import streamlit as st 

def query(query:str): 
    if not query.strip():
        st.error("Message cannot be empty.")
    else:
        st.markdown("good")