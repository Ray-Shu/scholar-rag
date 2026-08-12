import os 
from dotenv import load_dotenv 

from google import genai 
from google.genai import types 

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 

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

if __name__ == "__main__": 
    load_dotenv() # loads env vars 

    client = genai.Client()
    #semantic_example(client) 
    #controlling_embedding_size_example(client)

    


