import pymupdf
from pathlib import Path

PAPERS_FOLDER = Path.cwd() / "src/scholar-rag/papers/"
if __name__ == "__main__": 
    doc = pymupdf.open(PAPERS_FOLDER / "rtu.pdf")
    out = open("output.txt", "wb") 
    for page in doc: 
        text = page.get_text().encode("utf8")
        out.write(text)
        out.write(bytes((12,)))
    out.close()
