import streamlit as st 

def init_messages(): 
    if "messages" not in st.session_state: 
        st.session_state.messages = []

def render_messages(): 
    for msg in st.session_state.messages: 
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def add_message(role, content): 
    st.session_state.messages.append(
        {
            "role": "user", 
            "content": content
        }
    )