"""
A collection of utilities for local operations. For example, local storage. 
"""
import os 
from pathlib import Path
from PIL import Image

# local globals 
PAPERS_FOLDER = Path(__file__).resolve().parent / "papers"
STORAGE_PATH = Path.cwd() / "storage"

def store_pages(page_img:Image, paper_name:str, page_num:int): 
    """
    Stores pages in a local storage. 
    """
    folder_path = STORAGE_PATH / paper_name 

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
         
    page_img.save(folder_path / f"{page_num}.png")
    
