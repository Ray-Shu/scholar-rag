from fastapi import UploadFile, BackgroundTasks, APIRouter, Depends
import uuid 

from scholar_rag.api.dependencies import get_model, get_processor, get_progress
import scholar_rag.core.backend_storing as backend_storing


router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("")
async def upload(
    files: list[UploadFile], 
    background_task: BackgroundTasks,
    model = Depends(get_model),
    processor = Depends(get_processor),
    progress = Depends(get_progress)
):
    task_id = str(uuid.uuid4()) 
    progress[task_id] = {"status": "processing", "progress": 0.0}

    file_list = [(await file.read(), file.filename) for file in files]
    background_task.add_task(
        backend_storing.store_and_embed, 
        file_list,
        model,
        processor,
        task_id,
        progress
    )

    return {
            "message": f"{len(files)} are processing in the background",
            "task_id": task_id
        }

@router.get("/status/{task_id}")
async def get_file_upload_status(task_id: str, progress = Depends(get_progress)):
    return progress.get(task_id, {"status": "not_found", "progress": 0.0})

