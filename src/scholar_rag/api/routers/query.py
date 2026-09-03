from fastapi import UploadFile, BackgroundTasks, APIRouter, Depends
from pydantic import BaseModel
import uuid

from scholar_rag.api.dependencies import get_model, get_processor
import scholar_rag.core.querying as querying

class QueryRequest(BaseModel): 
    query: str

router = APIRouter(prefix="/query", tags=["query"])

@router.post("")
async def query(
    payload:QueryRequest,
    background_task: BackgroundTasks,
    model = Depends(get_model),
    processor = Depends(get_processor),
): 
    task_id = str(uuid.uuid4())
    background_task.add_task(
        querying.query, 
        payload.query, 
        model, 
        processor
    )
  



    return {
        "message": "Agent is thinking...",
        "task_id": task_id
    }