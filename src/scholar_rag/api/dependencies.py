from fastapi import Request

def get_model(request: Request):
    return request.app.state.model

def get_processor(request: Request): 
    return request.app.state.processor

def get_progress(request: Request) -> dict:
    return request.app.state.progress