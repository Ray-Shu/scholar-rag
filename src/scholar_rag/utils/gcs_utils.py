"""
A collection of functions dedicated to utilities for Google Cloud Storage.
"""

from google.cloud.storage import Client, transfer_manager
from pathlib import Path
from dotenv import load_dotenv 

def upload_many_blob_from_memory(transfer_manager, file_blob_pairs):
    """
    Uploads a list of contents concurrently for optimized upload speed. 

    Args: 
        
    """

    transfer_manager.upload_many(
        file_blob_pairs=file_blob_pairs,
        skip_if_exists=True, 
        worker_type=transfer_manager.THREAD
    )
    

def delete_blob(storage_client, bucket_name, blob_name): 




# if __name__ == "__main__": 
#     load_dotenv() # loads env vars 
#     source_dir = Path("./data")
#     upload_blob("scholar-rag-papers-bucket", source_file_name="storage/rtu/1.webp", destination_blob_name="hi")