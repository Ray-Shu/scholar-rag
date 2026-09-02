# scholar-rag

A lightweight RAG web app that utilizes the [ColPali](https://github.com/illuin-tech/colpali)  vision language model to embed files into batches of embedding vectors. An example of what I mean is shown here: 

![Credit: **ColPali: Efficient Document Retrieval with Vision Language Models**](assets/vlm_multisim.jpg)

The application uses FastAPI as the backend, and Streamlit as the frontend. Qdrant Cloud is used to store vector embeddings, and Google Cloud Storage is used to store the uploaded papers.