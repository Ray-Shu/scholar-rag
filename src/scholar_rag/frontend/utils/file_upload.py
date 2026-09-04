import streamlit as st 
import requests
import time

MAX_WAIT_SECONDS = 300 
POLLING_INTERVAL = 1.5

def upload(uploader):
    if uploader:
        try: 
            response = requests.post(
                "http://localhost:8000/upload/",
                files= [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploader],
                timeout = 10
            )
        except requests.RequestException as e:
            st.error(f"Request Error: {e}") 
            return

        task_id = response.json().get("task_id")
 
        if task_id: 
            st.info("Embedding documents...")
            progress_bar = st.progress(0)
            status_text = st.empty() 

            start_time = time.time() 

            while True: 
                # check for timeout 
                elapsed_time = time.time() - start_time 
                if elapsed_time >= MAX_WAIT_SECONDS: 
                    progress_bar.empty() 
                    st.error("Processing timed out.")
                    break

                # check for connection to backend 
                try: 
                    status_result = requests.get(
                        f"http://localhost:8000/upload/status/{task_id}",
                        timeout = 10
                        )
                except requests.RequestException as e: 
                    st.error(f"Failed to connect to backend: {e}")
                    break 

                if status_result.status_code == 200: 
                    data = status_result.json() 
                    progress_bar.progress(data["progress"])
                    status_text.text(f"Processing: {int(data['progress'] * 100)}%")

                    if data["status"] == "completed":
                        st.success("Files uploaded!")
                        break 
                    elif data["status"] == "failed": 
                        progress_bar.empty()
                        st.error("Upload failed.")
                        break
                    elif data["status"] == "not_found":
                        progress_bar.empty()
                        st.error("Task ID not found. Could not upload file.")
                else: 
                    st.error(f"Server responded with error code: {status_result.status_code}")
                    break

                time.sleep(POLLING_INTERVAL)

    else: 
        st.warning("No files uploaded.")


