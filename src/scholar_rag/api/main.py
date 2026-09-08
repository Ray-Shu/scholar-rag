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

@app.middleware("http")
async def add_security_headers(request: Request, call_next): 
    response = await call_next(request)

    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"

    if "Referrer-Policy" not in response.headers:
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    if request.url.hostname not in ("localhost", "127.0.0.1"):
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )
    return response

app.include_router(upload.router)
app.include_router(query.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

