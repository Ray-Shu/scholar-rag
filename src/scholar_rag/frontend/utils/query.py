import streamlit as st
import requests 
import time

MAX_WAIT_SECONDS = 300 
POLLING_INTERVAL = 1.5

def is_query_good(query:str): 
    """
    Processes a users query. Returns a 1 if the query is legitimate, else 0.
    """
    if not query.strip():
        st.error("Message cannot be empty.")
        return 0
    else:
        return 1

def query_agent(query:str):
    try:
        response = requests.post(
            "http://localhost:8000/query/", 
            json={"query": query},
            timeout=10
        )
    except Exception as e:
        st.error(f"Request could not be posted with error {e}.")
        return None

    task_id = response.json().get("task_id")

    if task_id: 
        status_placeholder = st.empty()
        status_placeholder.info(response.json().get("message"))
        start_time = time.time()
        try:
            while True: 
                elapsed_time = time.time() - start_time
                if elapsed_time >= MAX_WAIT_SECONDS:
                    st.error("Process timed out.")
                    break

                try: 
                    status_result = requests.get(
                        f"http://localhost:8000/query/status/{task_id}",
                        timeout = 10
                    )
                except requests.RequestException as e: 
                    st.error(f"Failed to retrieve agent response: {e}")
                    break

                if status_result.status_code == 200: 
                    data = status_result.json()
                    if data["status"] == "complete": 
                        agent_reply = data["output"]
                        return agent_reply
                    elif data["status"] == "not_found": 
                        st.error(f"Task ID not found")
                        break
                    elif data["status"] == "failed":
                        st.error(f"Agent failed to respond with error: {data.get('error', 'Unknown error')}")
                        break
                else: 
                    st.error(f"Server responded with error code: {status_result.status_code}")
                    break
                time.sleep(POLLING_INTERVAL)
        finally:
            status_placeholder.empty()
    return None



