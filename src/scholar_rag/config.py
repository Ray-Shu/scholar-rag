from pathlib import Path
from dotenv import load_dotenv 
load_dotenv()

PAPERS_FOLDER = Path(__file__).resolve().parent / "papers"
COLLECTION_NAME = "papers"

GCS_BUCKET_NAME = "scholar-rag-papers-bucket"
GCS_RESEARCH_PAPER_FOLDER_NAME = "research-papers"
IMAGE_FORMAT = "webp"

MODEL_NAME = "vidore/colqwen2-v1.0-hf"
GENERATION_MODEL = "gemini-3.6-flash"
EMBEDDING_SIZE = 128 # from colpali's documentation

SAVE_LOCAL = False
SAVE_CLOUD = True 