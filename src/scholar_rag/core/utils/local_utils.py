"""
A collection of utilities for local operations. For example, local storage. 
"""
import os 
from pathlib import Path
from PIL import Image

PAPERS_FOLDER = Path(__file__).resolve().parent / "papers"
STORAGE_PATH = Path.cwd() / "storage"

def get_storage_folder_size(): 
    total_bytes = sum(f.stat().st_size for f in STORAGE_PATH.rglob("*") if f.is_file()) 
    total_mb = total_bytes / (1024 * 1024)
    print(f"Size of storage folder is {total_mb} MB.")

def store_pages(page_img:Image, paper_name: str, relative_image_key:str, image_format:str = "webp"): 
    """
    Stores pages in a local storage. 
    Uses webp format for ~1/3 of file size of PNG formats at the sacrifice of some lossy compression. 
    """
    folder_path = STORAGE_PATH / paper_name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    page_img.save(STORAGE_PATH / f"{relative_image_key}", format=image_format, lossless=True)