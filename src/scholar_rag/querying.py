import base64
from pathlib import Path
from dotenv import load_dotenv 

import scholar_rag.utils.vlm_utils as vlm_utils 
import scholar_rag.utils.qdrant_utils as qdrant_utils

from google import genai

COLLECTION_NAME = "papers"
MODEL_NAME = "vidore/colqwen2-v1.0-hf"
DEVICE = vlm_utils.get_device()

def main(): 
    test_query = "What is an RTU?" 
    model, processor = vlm_utils.create_colqwen_model_and_processor(device=DEVICE, model_name=MODEL_NAME)
    embeddings = vlm_utils.embed_input(model=model, processor=processor, input_type="query", input=[test_query]) # returns [batch_size, tokens, embed_dim]

    client = qdrant_utils.get_client(path="./qdrant_data")
    result = qdrant_utils.query(client=client, query=embeddings[0], collection_name="papers")
    print(result.points[0].payload)

    # create prompt
    prompt = [
        {
            "type": "text",
            "text": f"You're given this query: {test_query}."
                    f"Use the data provided in `images` to give a correct and consice answer with sources supplemented."
                    f"Use the knowledge provided from the metadata context: {result}, and your own knowledge."
        }
    ]

    for i in range(len(result.points)): 
        image_path = Path(result.points[i].payload["image_path"])
        prompt.append( 
            {  
                "type": "image", 
                # api expects base64 but needs to serializable so use ascii decode to become string. 
                "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                "mime_type": "image/png"  # MIME type: multipurpose internet mail extension
            
            }
        )
    
    client = genai.Client()
    interaction = client.interactions.create(
        model="gemini-3.7-flash",
        input= prompt
    )

    print(interaction.output_text)
    
if __name__ == "__main__": 
    load_dotenv() # loads env vars
    main() 