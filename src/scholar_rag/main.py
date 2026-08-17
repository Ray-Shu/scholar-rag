import os 
from pathlib import Path
from dotenv import load_dotenv 

from google import genai 
from google.genai import types 

from transformers.models import ColPaliProcessor
from transformers.models import ColQwen2PreTrainedModel
from transformers.utils.import_utils import is_flash_attn_2_available 

import torch
import numpy as np
import pymupdf
import qdrant_client
from PIL import Image

# -- GLOBALS -- # 
CWD = Path.cwd() / "src/scholar_rag"
PAPERS_FOLDER = CWD / "papers"

DOC_NAME = "grouped-query-attention"
DOC = DOC_NAME + ".pdf"

EMBEDDING_DIM = 768

MODEL_NAME = "vidore/colqwen2-v1.0"

DEVICE = "mps"

def main(): 
    model = ColQwen2PreTrainedModel.from_pretrained( 
        MODEL_NAME, 
        torch_dtype = torch.bfloat16,
        device_map = DEVICE, 
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available else None
    ).eval()

    print(model)

    # document = pymupdf.open(filename=PAPERS_FOLDER / DOC)

    # image_batch = [] 
    # metadata_batch = []
    # for page_num, page in enumerate(document): 
    #     pix = page.get_pixmap(dpi=150)
    #     page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    #     image_batch.append(page_img)
    #     metadata_batch.append(
    #         {
    #             "document": DOC_NAME,
    #             "page_number": page_num
    #         }
    #     )

    

if __name__ == "__main__": 
    load_dotenv() # loads env vars 
    main() 
    


