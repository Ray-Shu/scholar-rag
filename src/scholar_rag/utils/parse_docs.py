import pymupdf
import pymupdf4llm
from pathlib import Path

from scholar_rag.utils.chunking import fixed_size_chunking

CWD = Path.cwd() / "src/scholar_rag"
PAPERS_FOLDER = CWD / "papers"
OUTPUT_FOLDER = CWD / "output"

def parse_to_markdown(doc_name:str, doc_type:str): 
    """
    Parses a pdf document to markdown given a document name and type. 
    Ex: 
        doc_name = 'paper' 
        doc_type = 'pdf'
    """
    doc_file = doc_name + "." + doc_type
    md = pymupdf4llm.to_markdown(PAPERS_FOLDER / doc_file)

    return md 
