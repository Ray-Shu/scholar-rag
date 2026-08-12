import os 
from dotenv import load_dotenv 

from google import genai 
from google.genai import types 

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 

from scholar_rag.utils import chunking 
from scholar_rag.utils import parse_docs 
from scholar_rag.utils import embed

def semantic_example(client): 
    texts = [
        "What is the meaning of life?",
        "What is the purpose of existence?",
        "How do I bake a cake?",
    ]
    
    result = client.models.embed_content(
        model = "gemini-embedding-001", 
        contents=texts, 
        config = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )

    df = pd.DataFrame( 
        cosine_similarity([e.values for e in result.embeddings]),
        index=texts,
        columns=texts
    )

    print(df)

def controlling_embedding_size_example(client): 
    result = client.models.embed_content(
        model = "gemini-embedding-2",
        contents = "What is the meaning of life?",
        config = types.EmbedContentConfig(output_dimensionality=768)
    )

    [embedding_obj] = result.embeddings

    print(embedding_obj)
    embedding_length = len(embedding_obj.values)
    print(embedding_length)

def main(): 
    doc_name = "rtu"
    doc_type = "pdf"
    document = parse_docs.parse_to_markdown(doc_name=doc_name, doc_type=doc_type)

    chunked_doc = chunking.fixed_size_chunking(document)

    res = embed.create_embeddings(document)

    print(res)
    

if __name__ == "__main__": 
    load_dotenv() # loads env vars 
    main() 

    #client = genai.Client()
    #semantic_example(client) 
    #controlling_embedding_size_example(client)

    


