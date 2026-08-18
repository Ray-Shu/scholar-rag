import os 
from pathlib import Path
from dotenv import load_dotenv 

from google import genai 
from google.genai import types 

from transformers.models import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available 

import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, MultiVectorComparator, MultiVectorConfig

import torch
import numpy as np
import pymupdf
from PIL import Image

# -- GLOBALS -- # 
CWD = Path.cwd() / "src/scholar_rag"
PAPERS_FOLDER = CWD / "papers"

COLLECTION_NAME = "papers"
DOC_NAME = "grouped-query-attention"
DOC = DOC_NAME + ".pdf"

BATCH_SIZE = 8

MODEL_NAME = "vidore/colqwen2-v1.0-hf"
EMBEDDING_SIZE = 128 # from colpali's documentation

DEVICE = "cuda"

# ------------- # 

def create_qdrant_client(path, collection_name): 
    client = qdrant_client.QdrantClient(path=path)

    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection( 
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE, 
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        )
    )

    return client 

def create_colqwen_model_and_processor(): 
    # Model download
    if DEVICE == "cuda" and is_flash_attn_2_available(): 
        attn_impl = "flash_attention_2"
    else: 
        attn_impl = "sdpa" 

    model = ColQwen2ForRetrieval.from_pretrained( 
        MODEL_NAME, 
        torch_dtype = torch.bfloat16,
        device_map = DEVICE, 
        attn_implementation = attn_impl
    ).eval()

    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)

    return model, processor 

def main(): 
    client = create_qdrant_client(path="./qdrant_data", collection_name=COLLECTION_NAME)
    model, processor = create_colqwen_model_and_processor()

    # Parse documents
    points_id = 0
    document = pymupdf.open(filename=PAPERS_FOLDER / DOC)
    image_batch = [] 
    metadata_batch = []
    for page_num, page in enumerate(document):  # iterate over document pages 
        pix = page.get_pixmap(dpi=150)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        image_batch.append(page_img)
        metadata_batch.append(
            {
                "document": DOC_NAME,
                "page_number": page_num
            }
        )

        if len(image_batch) >= BATCH_SIZE or len(image_batch) >= len(document): 
            # run embedding logic 
            processed_images = processor.process_images(image_batch).to(model.device)
            with torch.no_grad(): 
                out = model(**processed_images)  # torch.Size([BATCH_SIZE, 747, 128])

            batch_embeddings = out.embeddings.cpu().float().numpy().tolist()

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
    


