import faiss 
import numpy as np 

def build_index(embeddings, embedding_dim): 
    index = faiss.IndexFlatIP(d=embedding_dim)
    
