import streamlit as st 
import requests

left_column, right_column = st.columns(2)

uploader = left_column.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

if st.button("Upload files"): 
    if uploader:
        requests.post(
            "http://localhost:8000/upload",
            files= [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploader]
        )
    else: 
        st.markdown("No files uploaded.")
         #@ "file": (file.name, file.getvalue(), "application/pdf")