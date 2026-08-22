import torch

from transformers.models import ColQwen2ForRetrieval, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available 

def create_colqwen_model_and_processor(device:str, model_name:str): 
    """
    Downloads the Colqwen processor and model to use as the encoder. 
    """
    if device == "cuda" and is_flash_attn_2_available(): 
        attn_impl = "flash_attention_2"
    else: 
        attn_impl = "sdpa" 

    model = ColQwen2ForRetrieval.from_pretrained( 
        model_name, 
        torch_dtype = torch.bfloat16,
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
        embeddings: The embedded vector of the image.
    """ 

    if input_type == "image": 
        processed_images = processor.process_images(input).to(model.device)
        with torch.no_grad(): 
            out = model(**processed_images)  # torch.Size([BATCH_SIZE, 747, 128])
    elif input_type == "query": 
        processed_queries = processor.process_queries(input).to(model.device)
        with torch.no_grad(): 
            out = model(**processed_queries)  # torch.Size([BATCH_SIZE, 747, 128])
    else: 
        assert ValueError(f"Invalid input type {input_type}.")

    embeddings = out.embeddings.cpu().float().numpy().tolist() 

    return embeddings