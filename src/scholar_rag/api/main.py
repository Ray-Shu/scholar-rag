from dotenv import load_dotenv 
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .routers import upload
from .routers import query

import logging

import scholar_rag.core.utils.vlm_utils as vlm_utils
from scholar_rag import config 

# lifespan context manager 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Colqwen model into GPU.")
    device = vlm_utils.get_device()
    model, processor = vlm_utils.create_colqwen_model_and_processor(device, model_name=config.MODEL_NAME)

    # store model into app's state
    app.state.model = model 
    app.state.processor = processor 
    print("Model loaded successfully.")

    # query storage (in RAM, no DB)
    app.state.query_results = {}

    # progress bars for pdf downloads (in-memory RAM)
    app.state.progress = {}

    # pauses lifespan function and runs webservice 
    yield   

    # shutdown logic 
    print("Shutting down server and clearing model")
    app.state.model = None 
    app.state.processor = None 

app = FastAPI(lifespan=lifespan)

@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logging.exception(f"Error on {request.url.path}")
    return JSONResponse(
        status_code = 500,
        content = {"detail": "Internal server error"}
    )

app.include_router(upload.router)
app.include_router(query.router)

