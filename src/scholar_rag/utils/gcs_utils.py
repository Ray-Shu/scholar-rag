"""
A collection of functions dedicated to utilities for Google Cloud Storage.
"""

from google.cloud.storage import transfer_manager, Client
from google.api_core.exceptions import PreconditionFailed
from pathlib import Path
from dotenv import load_dotenv 

from google.genai import types

def upload_blob_from_disk(bucket, source_file_name, destination_blob_name):
    """
    Uploads a file object to GCS from local disk (NOT from memory). 
    """
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def upload_many_blob_from_memory(file_blob_pairs):
    """
    Uploads a list of contents concurrently for optimized upload speed. 

    Args: 
        transfer_manager: The GCS transfer_manager object 
        file_blob_pairs: A list of tuples of a file (or filename) and blob object. The file can be an IOBase class or str. 
    """
    results = transfer_manager.upload_many(
        file_blob_pairs=file_blob_pairs,
        skip_if_exists=True, 
        worker_type=transfer_manager.THREAD
    )

    actual_failures = [
        r for r in results
        if isinstance(r, Exception) and not isinstance(r, PreconditionFailed)
    ]

    assert not actual_failures, f"Upload had unexpected failures: {actual_failures}"
    

def delete_blob(bucket, blob_name): 
    blob = bucket.blob(blob_name)
    blob.delete()

def download_blob_into_memory(bucket, blob_name): 
    blob = bucket.blob(blob_name)
    contents = blob.download_as_bytes()
    return contents 


if __name__ == "__main__": 
    load_dotenv() # loads env vars 

    client = Client()
    bucket = client.bucket("scholar-rag-papers-bucket")

    content = download_blob_into_memory(bucket, blob_name="rtu/001.webp")

    image_part = types.Part.from_bytes( 
        data=content,
        mime_type="image/webp"
    )

    print(image_part)
    
    