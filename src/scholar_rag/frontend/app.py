import streamlit as st 
import requests
import time

import scholar_rag.frontend.utils.file_upload as file_upload


st.markdown(
    """
    <style>
    .st-key-floating_uploader {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
        width: 320px;
        z-index: 999;
        background-color: rgb(14, 17, 23);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(250, 250, 250, 0.2);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

left_column, right_column = st.columns([2, 1])

# pdf uploader 
with st.container(key="floating_uploader"):
    uploader = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    if st.button("Upload files"):
        file_upload.upload(uploader=uploader)

# chat 
if prompt := st.chat_input("Enter a message."):
    with st.chat_message("user"):
        st.markdown(prompt)