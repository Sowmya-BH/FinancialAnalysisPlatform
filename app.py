
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import os

from crew import CkdV3   # your CrewAI class

app = FastAPI(title="Financial Document Analysis API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/analyze")
async def analyze_pdf(pdf: UploadFile = File(...), input_field: str = "Total gross profit"):

    # 1. Validate PDF
    if not pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 2. Save uploaded file
    pdf_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{pdf_id}.pdf"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf.file, buffer)

    print(f"📄 Saved PDF at {pdf_path}")

    # 3. Prepare inputs for CrewAI pipeline
    inputs = {
        "pdf_path": str(pdf_path),
        "input_field": input_field
    }

    try:
        print("🚀 Running CrewAI pipeline…")
        result = CkdV3().crew().kickoff(inputs=inputs)

        return {
            "status": "success",
            "pdf_id": pdf_id,
            "result": result  # Agent output
        }

    except Exception as e:
        print("❌ CrewAI Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "OK"}
