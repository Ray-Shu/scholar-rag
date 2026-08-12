from google import genai 
from google.genai import types 

MAX_BATCH_SIZE = 100 

def create_embeddings(document): 
    """
    Creates embeddings given a chunked document.

    Returns: 
        The embeddings of the chunked documents. 
    """

    client = genai.Client() 

    embeddings = []
    for i in range(0, len(document), MAX_BATCH_SIZE): 
        batch = document[i:i + MAX_BATCH_SIZE]
        result = client.models.embed_content(
            model = "gemini-embedding-001", 
            contents = [
                types.Content(parts = [types.Part.from_text(text=chunk)]) for chunk in batch
            ]
        )

        embeddings.extend(result.embeddings)

    return embeddings