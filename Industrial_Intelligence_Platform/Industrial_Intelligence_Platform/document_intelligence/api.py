"""FastAPI REST service: POST /extract with a DOCX/PDF/image upload."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from extractor import MetadataExtractor

app = FastAPI(title="DocuLens AI API", version="1.0.0",
              description="Extract six rental-agreement metadata fields from DOCX/PDF/images.")
extractor = MetadataExtractor()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".docx", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        raise HTTPException(415, "Upload a DOCX, PDF, or image file.")
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            path = Path(tmp.name)
        result = extractor.extract_file(path)
        path.unlink(missing_ok=True)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
