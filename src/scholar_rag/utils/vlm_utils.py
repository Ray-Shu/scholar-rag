import os

import torch

from transformers.models import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available 

def get_device(device:str=None): 
    """
    Picks the best available accelerator so the same code runs on a CUDA box and a Mac. 

    Args: 
        device: An explicit device string that overrides detection. Falls back to the 
            SCHOLAR_RAG_DEVICE env var, then to autodetection. 

    Returns: 
        device: One of 'cuda', 'mps' or 'cpu'.
    """
    device = device or os.getenv("SCHOLAR_RAG_DEVICE")
    if device: 
        return device
    if torch.cuda.is_available(): 
        return "cuda"
    if torch.backends.mps.is_available(): 
        return "mps"
    return "cpu"

def get_dtype(device:str): 
    """
    Returns the dtype to load the model in. CPU has no fast bfloat16 kernels, so it stays in float32.
    """
    return torch.float32 if device == "cpu" else torch.bfloat16

def create_colqwen_model_and_processor(device:str=None, model_name:str="vidore/colqwen2-v1.0-hf"): 
    """
    Downloads the Colqwen processor and model to use as the encoder. 
    """
    device = get_device(device)

    if device == "cuda" and is_flash_attn_2_available(): 
        attn_impl = "flash_attention_2"
    else: 
        attn_impl = "sdpa" 

    model = ColQwen2ForRetrieval.from_pretrained( 
        model_name, 
        dtype = get_dtype(device),
        device_map = device, 
        attn_implementation = attn_impl
    ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)

    return model, processor 

def embed_input(model, processor, input_type, input):
    """
    Encodes an input. 

    Args: 
        model: The VLM model. 
        processor: The VLM processor. 
        input_type: A string 'image' or 'query'.
        input: An individual or batched input. 

    Returns: 
        embeddings: The embedded vector of the input.
    """ 

    if input_type == "image": 
        processed_images = processor.process_images(input).to(model.device)
        with torch.no_grad(): 
            out = model(**processed_images)  # torch.Size([BATCH_SIZE, 747, 128])
    elif input_type == "query": 
        processed_queries = processor.process_queries(input).to(model.device)
        with torch.no_grad(): 
            out = model(**processed_queries)
    else: 
        assert ValueError(f"Invalid input type {input_type}.")

    embeddings = out.embeddings.cpu().float().numpy().tolist() 

    return embeddings