import os 
from dotenv import load_dotenv 

from google import genai 
from google.genai import types 

# vector db 
import faiss 

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 

from scholar_rag.utils import chunking 
from scholar_rag.utils import parse_docs 
from scholar_rag.utils import embed


DOC_NAME = "grouped-query-attention"
DOC_TYPE = "pdf"

EMBEDDING_DIM = 768

def main(): 
    # parse document to markdown format
    document = parse_docs.parse_to_markdown(doc_name=DOC_NAME, doc_type=DOC_TYPE)

    # chunk document
    chunked_doc = chunking.fixed_size_chunking(document)
    print(chunked_doc[3])

    # create embedding vectors
    #embeddings = embed.create_embeddings(chunked_doc, embedding_dim=EMBEDDING_DIM)

    # store embedding vectors 
    


    

if __name__ == "__main__": 
    load_dotenv() # loads env vars 
    main() 

    #client = genai.Client()
    #semantic_example(client) 
    #controlling_embedding_size_example(client)

    


