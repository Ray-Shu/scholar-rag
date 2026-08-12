import pymupdf
import pymupdf4llm
from pathlib import Path

from scholar_rag.utils.chunking import fixed_size_chunking

CWD = Path.cwd() / "src/scholar_rag"
PAPERS_FOLDER = CWD / "papers"
OUTPUT_FOLDER = CWD / "output"

if __name__ == "__main__": 
    md = pymupdf4llm.to_markdown(PAPERS_FOLDER / "rtu.pdf")

    document = fixed_size_chunking(md)

    with open(OUTPUT_FOLDER / "output_text.txt", "wb") as out_text: 
        out_text.write(document)
    out_text.close()






    # with open(OUTPUT_FOLDER / "output_text.txt", "wb") as out_text: 
    #     for page_num, page in enumerate(doc, start=1): 
    #         # extract text 
    #         text = page.get_text().encode("utf8")
    #         out_text.write(text)
    #         out_text.write(bytes((12,)))

    #         # extract page imgs 
    #         pix = page.get_pixmap()
    #         img_path = OUTPUT_FOLDER / f"output_img_{page_num}.png"
    #         pix.save(img_path)

    # doc.close()


