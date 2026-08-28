import os 
import io
from pathlib import Path
from dotenv import load_dotenv 

from qdrant_client.models import PointStruct

from google.cloud.storage import Client

import scholar_rag.utils.qdrant_utils as qdrant_utils
import scholar_rag.utils.vlm_utils as vlm_utils
import scholar_rag.utils.local_utils as local_utils
import scholar_rag.utils.gcs_utils as gcs_utils
from scholar_rag import config

import pymupdf
from PIL import Image

BATCH_SIZE = 8
DEVICE = vlm_utils.get_device()

def main(): 
    assert config.PAPERS_FOLDER.exists(), f"Folder {config.PAPERS_FOLDER} doesn't exist."

    storage_client = Client() 
    bucket = storage_client.bucket(bucket_name=config.GCS_BUCKET_NAME)

    client = qdrant_utils.get_client()
    qdrant_utils.create_collection(client=client, collection_name=config.COLLECTION_NAME, embedding_size=config.EMBEDDING_SIZE)
    model, processor = vlm_utils.create_colqwen_model_and_processor(DEVICE, config.MODEL_NAME)

    points_id = 0
    image_batch = [] 
    metadata_batch = []
    file_blob_pairs = [] 

    # iterate over documents 
    for papers in config.PAPERS_FOLDER.glob("*.pdf"): 
        document = pymupdf.open(filename=config.PAPERS_FOLDER / papers.name) 
        paper_name = papers.name[:-4]

        for page_num, page in enumerate(document):  # iterate over document pages 
            # create image of the page 
            pix = page.get_pixmap(dpi=150)
            page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            relative_image_key = f"{paper_name}/{page_num:03d}.{config.IMAGE_FORMAT}"

            # saving paper .png files onto local disk
            if config.SAVE_LOCAL:
                local_utils.store_pages(page_img, paper_name, relative_image_key, image_format=config.IMAGE_FORMAT)

            buffer = io.BytesIO()
            page_img.save(buffer, format=config.IMAGE_FORMAT)
            buffer.seek(0)
            blob = bucket.blob(blob_name=relative_image_key)
            
            # batching 
            image_batch.append(page_img)
            metadata_batch.append(
                {
                    "document": paper_name,
                    "page_number": page_num,
                    "relative_image_key": relative_image_key
                }
            )
            file_blob_pairs.append((buffer, blob))

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
                    collection_name=config.COLLECTION_NAME,
                    points = points
                )

                image_batch = []
                metadata_batch = []

    gcs_utils.upload_many_blob_from_memory(file_blob_pairs=file_blob_pairs)
                

if __name__ == "__main__": 
    load_dotenv() # loads env vars 
    main() 
    


