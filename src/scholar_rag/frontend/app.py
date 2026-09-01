import streamlit as st 
import requests
import time

import scholar_rag.frontend.utils.file_upload as file_upload


left_column, right_column = st.columns([2, 1])

with right_column.container(): 
    uploader = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    if st.button("Upload files"):
        file_upload.upload(uploader=uploader)