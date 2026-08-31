from dotenv import load_dotenv 
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, BackgroundTasks

import uuid

import scholar_rag.core.backend_storing as backend_storing
import scholar_rag.core.utils.vlm_utils as vlm_utils
from scholar_rag import config 

# lifespan context manager 
async def lifespan(app: FastAPI):
    print("Loading Colqwen model into GPU.")
    device = vlm_utils.get_device()
    model, processor = vlm_utils.create_colqwen_model_and_processor(device, model_name=config.MODEL_NAME)

    # store model into app's state
    app.state.model = model 
    app.state.processor = processor 
    print("Model loaded successfully.")

    # progress bars for pdf downloads (in-memory RAM)
    app.state.progress = {}

    # pauses lifespan function and runs webservice 
    yield   

    # shutdown logic 
    print("Shutting down server and clearing model")
    app.state.model = None 
    app.state.processor = None 

app = FastAPI(lifespan=lifespan) 

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.post("/upload")
async def upload(files: list[UploadFile], background_task: BackgroundTasks):
    task_id = str(uuid.uuid4) 
    app.state.progress[task_id] = {"status": "processing", "progress": 0.0}

    file_list = [(file.file.read(), file.filename) for file in files]
    background_task.add_task(
        backend_storing.store_and_embed, 
        file_list,
        app.state.model,
        app.state.processor,
        task_id,
        app.state.progress
    )

    return {
            "message": f"{len(files)} are processing in the background",
            "task_id": task_id
        }

@app.get("/status/{task_id}")
async def get_file_upload_status(task_id: str):
    return app.state.progress.get(task_id, {"status": "not_found", "progress": 0.0})

