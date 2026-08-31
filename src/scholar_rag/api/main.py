from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI() 

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.post("/upload")
async def upload(files: list[UploadFile]): 
    for file in files:
        print(file.filename)
        # pdf_bytes = await file.read() 
        # print(pdf_bytes)
