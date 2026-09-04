# scholar-rag

A lightweight RAG web app that utilizes the [ColPali](https://github.com/illuin-tech/colpali) vision language model to embed files into batches of embedding vectors. An example is shown here: 

<p align="center">
  <img src="assets/vlm_multisim.jpg" alt="Credit: ColPali: Efficient Document Retrieval with Vision Language Models" style="width: 75%;">
</p>

After looking into text-chunking strategies, I noticed two things: 
1. They can't parse mathematical formulations well.
2. It requires more complexity to render images and graphs in files. 

So I decided to look into other methods and found VLMs, specifically the ColPali paper. Upon experimenting with inputting math-heavy papers, I noticed that responses don't make many (if any) mistakes.

I created this app because of two things. 1) upon becoming a research assistant, I've had to read many papers to gauge modern SOTA models, experiments, techniques, etc. I needed a way to reference existing ideas without having to parse through my library of papers, and be able to get smarter responses. 2) I wanted to learn about web application and concepts related to ML Engineer roles. 

---
### Implementation

The application uses FastAPI as the backend, and Streamlit as the frontend. Qdrant Cloud is used to store vector embeddings, and Google Cloud Storage is used to store the uploaded papers.

---
### To-Dos

- [ ] Add database
- [ ] Setup CI/CD Pipeline
- [ ] Setup tests
- [ ] Setup Docker
- [ ] Host on Google Cloud Run 
