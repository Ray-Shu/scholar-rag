import streamlit as st 

import scholar_rag.frontend.utils.file_upload as file_upload
import scholar_rag.frontend.utils.app_notes as app_notes

left_column, right_column = st.bottom.columns([0.8, 0.2])

with left_column:
    prompt = st.chat_input(
        "Enter a message.", 
    )
with right_column: 
    with st.popover("Upload PDF"):
        uploader = st.file_uploader(" ", type="pdf", accept_multiple_files=True)
        if st.button("Upload Files"):
            file_upload.upload(uploader=uploader)

if "messages" not in st.session_state: 
    st.session_state.messages = []

for msg in st.session_state.messages: 
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt: 
        with st.chat_message("user"): 
            st.markdown(prompt)
        st.session_state.messages.append(
            {
                "role": "user", 
                "content": prompt
            }
        )

with st.sidebar:
    st.title("App Notes")
    app_notes.get_app_notes()