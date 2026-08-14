import time

from google import genai 
from google.genai import types 

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.genai.errors import ClientError 

import numpy as np

MAX_BATCH_SIZE = 50 

# retries calling api after client error failure
@retry(
        retry=retry_if_exception_type(ClientError),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        stop=stop_after_attempt(6)
)
def _embed_batch(client, batch, embedding_dim): 
    result = client.models.embed_content(
                model = "gemini-embedding-001", 
                contents = [
                    types.Content(parts = [types.Part.from_text(text=chunk.page_content)]) for chunk in batch
                ],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", 
                    output_dimensionality=embedding_dim)
            )
    return result

def create_embeddings(document, embedding_dim=768): 
    """
    Creates embeddings given a chunked document.

    Returns: 
        A list of embeddings (the values) of the chunked documents. 
    """
    client = genai.Client() 

    embeddings = np.empty(shape=(len(document), embedding_dim), dtype=np.float32)
    for i in range(0, len(document), MAX_BATCH_SIZE): 
        batch = document[i:i + MAX_BATCH_SIZE]
        result = _embed_batch(client, batch, embedding_dim)

        # save embedding values into a matrix of shape [batch, embedding_dim]
        embeddings[i:i+MAX_BATCH_SIZE] = [e.values for e in result.embeddings]

        time.sleep(45) # stays under 100 req/min 

    return embeddings