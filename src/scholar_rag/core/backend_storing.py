import io

from qdrant_client.models import PointStruct

from google.cloud.storage import Client

import scholar_rag.core.utils.qdrant_utils as qdrant_utils
import scholar_rag.core.utils.vlm_utils as vlm_utils
import scholar_rag.core.utils.local_utils as local_utils
import scholar_rag.core.utils.gcs_utils as gcs_utils
from scholar_rag import config

import pymupdf
from PIL import Image
import uuid 

BATCH_SIZE = 8
DEVICE = vlm_utils.get_device()

def batch_and_store_embeddings(qdrant_client, model, processor, image_batch, metadata_batch): 
    """
    Stores vector embeddings into qdrant cloud.
    """
    batch_embeddings = vlm_utils.embed_input(model, processor, "image", image_batch)
    points = []
    for i, embedding in enumerate(batch_embeddings): 
        # create a unique ID for each embedding
        unique_str = metadata_batch[i]["relative_image_key"]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding, 
                payload=metadata_batch[i]
            )
        )

    qdrant_client.upsert(
        collection_name=config.COLLECTION_NAME,
        points = points
    )


def store_and_embed(files: list[tuple[bytes, str]], model, processor, task_id:str, progress_dict:dict): 
    """
    Parses through files (PDF's). \ 
    Stores the images of research papers into Google Cloud Storage. \
    Stores the vector embeddings from a VLM into Qdrant Cloud. 

    Args: 
        files: A list of tuples (file_bytes, file_name)
        model: VLM model
        processor: VLM processor
        task_id: A unique ID associated with an upload 
        pr
    """

    storage_client = Client() 
    bucket = storage_client.bucket(bucket_name=config.GCS_BUCKET_NAME)

    qdrant_client = qdrant_utils.get_client()
    qdrant_utils.create_collection(
        client=qdrant_client, 
        collection_name=config.COLLECTION_NAME, 
        embedding_size=config.EMBEDDING_SIZE
    )

    image_batch = [] 
    metadata_batch = []
    file_blob_pairs = [] 

    processed_pages = 0
    total_pages = 0 
    documents = []
    for file_bytes, filename in files: 
        document = pymupdf.open(stream=file_bytes, filetype="pdf") 
        total_pages += len(document)
        documents.append((document, filename))

    try:
        # iterate over documents 
        for file_bytes, filename in files: 
            document = pymupdf.open(stream=file_bytes, filetype="pdf") 
            paper_name = filename.rsplit(".", 1)[0]

            for page_num, page in enumerate(document):  # iterate over document pages 
                try:
                    # create image of the page 
                    pix = page.get_pixmap(dpi=150)
                    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except Exception as e: 
                    print("Skipping corrupted page {page_num} in {paper_name}: {e}")
                    continue

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

                if len(image_batch) >= BATCH_SIZE:
                    batch_and_store_embeddings(qdrant_client, model, processor, image_batch, metadata_batch)
                    
                    image_batch = []
                    metadata_batch = []

                processed_pages += 1
                percent_processed = processed_pages / (total_pages + 0.05)
                progress_dict[task_id]["progress"] = percent_processed


        if len(image_batch) >= 0: 
            batch_and_store_embeddings(qdrant_client, model, processor, image_batch, metadata_batch)

        progress_dict[task_id]["status"] = "completed"
        progress_dict[task_id]["progress"] = 1.0
        gcs_utils.upload_many_blob_from_memory(file_blob_pairs=file_blob_pairs)

    except Exception as e: 
        progress_dict[task_id]["status"] = "failed"
        progress_dict[task_id]["progress"] = 0.0
                

if __name__ == "__main__": 
    # load_dotenv() # loads env vars 
    pass

    


