from fastapi import UploadFile, BackgroundTasks, APIRouter, Depends
from pydantic import BaseModel
import uuid

from scholar_rag.api.dependencies import get_model, get_processor, get_query_results
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
    query_results = Depends(get_query_results)
): 
    task_id = str(uuid.uuid4())
    query_results[task_id] = {"status": "processing", "output": ""}
    background_task.add_task(
        querying.query, 
        payload.query, 
        model, 
        processor,
        query_results
    )
  
    return {
        "message": "Agent is thinking...",
        "task_id": task_id
    }

@router.get("/status/{task_id}")
async def get_agent_response(task_id: str, query_results = Depends(get_query_results)):
    return query_results.get(task_id, {"status": "failed", "output": ""})
