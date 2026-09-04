import streamlit as st 

import scholar_rag.frontend.utils.messages as messages
import scholar_rag.frontend.utils.file_upload as file_upload
import scholar_rag.frontend.utils.query as query
import scholar_rag.frontend.utils.app_notes as app_notes

messages.init_messages()
left_column, right_column = st.bottom.columns([0.8, 0.2])

with left_column:
    prompt = st.chat_input("Enter a message.")

    if prompt: 
        res = query.is_query_good(prompt)
        if res: 
            messages.add_message(role="user", content=prompt)
            agent_reply = query.query_agent(query=prompt)

        if agent_reply: 
            messages.add_message(role="assistant", content=agent_reply)

with right_column: 
    with st.popover("Upload PDF"):
        uploader = st.file_uploader(" ", type="pdf", accept_multiple_files=True)
        if st.button("Upload Files"):
            file_upload.upload(uploader=uploader)

messages.render_messages()


with st.sidebar:
    st.title("App Notes")
    app_notes.get_app_notes()