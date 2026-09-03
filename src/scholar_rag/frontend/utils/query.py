import streamlit as st
import requests 

def query(query:str): 
    """
    Processes a users query. Returns a 1 if the query is legitimate, else 0.
    """
    if not query.strip():
        st.error("Message cannot be empty.")
        return 0
    else:
        try:
            response = requests.post(
                "http://localhost:8000/query/", 
                json={"query": query}
            )
        except Exception as e:
            st.error(f"Request could not be posted with error {e}.")
            return 0

        if response.status_code == 200: 
            return 1
        else:
            st.error(f"Failed with status {response.status_code}: {response.text}")
            return 0 