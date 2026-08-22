import os 
from pathlib import Path
from dotenv import load_dotenv 

from google import genai 
from google.genai import types 

from qdrant_client.models import PointStruct
import scholar_rag.utils.qdrant_utils as qdrant_utils

import scholar_rag.utils.vlm_utils as vlm_utils

import torch
import numpy as np
import pymupdf
from PIL import Image

# -- GLOBALS -- # 
PAPERS_FOLDER = Path(__file__).resolve().parent / "papers"

COLLECTION_NAME = "papers"

BATCH_SIZE = 8

MODEL_NAME = "vidore/colqwen2-v1.0-hf"
EMBEDDING_SIZE = 128 # from colpali's documentation

DEVICE = "cuda"

# ------------- # 


def main(): 
    client = qdrant_utils.create_qdrant_client(path="./qdrant_data", collection_name=COLLECTION_NAME, embedding_size=EMBEDDING_SIZE)
    model, processor = vlm_utils.create_colqwen_model_and_processor(DEVICE, MODEL_NAME)

    points_id = 0
    image_batch = [] 
    metadata_batch = []

    if not PAPERS_FOLDER.exists():
       print(f"Folder {PAPERS_FOLDER} doesn't exist.") 
       return

    # iterate over documents 
    for papers in PAPERS_FOLDER.glob("*.pdf"): 
        document = pymupdf.open(filename=PAPERS_FOLDER / papers.name) 

        for page_num, page in enumerate(document):  # iterate over document pages 
            pix = page.get_pixmap(dpi=150)
            page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            image_batch.append(page_img)
            metadata_batch.append(
                {
                    "document": papers.name[:-4],
                    "page_number": page_num
                }
            )

            if len(image_batch) >= BATCH_SIZE or len(image_batch) >= len(document): 
                # run embedding logic 
                batch_embeddings = vlm_utils.embed_input(model, processor, "image", image_batch)

                points = []
                for i, embedding in enumerate(batch_embeddings): 
                    points.append(
                        PointStruct(
                            id=points_id,
                            vector=embedding, 
                            payload=metadata_batch[i]
                        )
                    )
                    points_id += 1

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points = points
                )

                image_batch = []
                metadata_batch = []
                

if __name__ == "__main__": 
    load_dotenv() # loads env vars 
    main() 
    


