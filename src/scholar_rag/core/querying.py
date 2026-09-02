import base64
from dotenv import load_dotenv 

import scholar_rag.core.utils.vlm_utils as vlm_utils 
import scholar_rag.core.utils.qdrant_utils as qdrant_utils
import scholar_rag.core.utils.gcs_utils as gcs_utils
from scholar_rag import config

from google import genai
from google.cloud.storage import Client

DEVICE = vlm_utils.get_device()

def query(query:str): 
    storage_client = Client() 
    bucket = storage_client.bucket(bucket_name=config.GCS_BUCKET_NAME)
    
    model, processor = vlm_utils.create_colqwen_model_and_processor(device=DEVICE, model_name=config.MODEL_NAME)
    embeddings = vlm_utils.embed_input(model=model, processor=processor, input_type="query", input=[test_query]) # returns [batch_size, tokens, embed_dim]

    client = qdrant_utils.get_client()
    result = qdrant_utils.query(client=client, query=embeddings[0], collection_name=config.COLLECTION_NAME)

    # create prompt
    prompt = [
        {
            "type": "text",
            "text": f"You're given this query: {query}."
                    f"Use the data provided in `images` to give a correct and consice answer with sources supplemented."
                    f"Use the knowledge provided from the metadata context: {result}, and your own knowledge."
        }
    ]

    for i in range(len(result.points)): 
        relative_image_key = result.points[i].payload["relative_image_key"]
        content = gcs_utils.download_blob_into_memory(bucket, blob_name=relative_image_key)

        prompt.append( 
            {  
                "type": "image", 
                # api expects base64 but needs to serializable so use ascii decode to become string. 
                "data": base64.b64encode(content).decode("ascii"),
                "mime_type": "image/webp"  # MIME type: multipurpose internet mail extension
            
            }
        )
    
    gemini_client = genai.Client()
    interaction = gemini_client.interactions.create(
        model=config.GENERATION_MODEL,
        input= prompt
    )

    return interaction.output_text
    
if __name__ == "__main__": 
    load_dotenv() # loads env vars
    query() 