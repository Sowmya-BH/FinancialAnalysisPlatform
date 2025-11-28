import pymupdf as fitz
import re
import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv
import os

from pymongo import MongoClient, UpdateOne
import os

SECTION_REGEX = r"(?P<header>^[A-Z][A-Za-z0-9 ,\-()]{3,}$)"
# SECTION_REGEX = r"(?P<header>^[A-Z][A-Za-z0-9 ,\-()]{3,}$)"
MONGO_URI = os.getenv("MONGO_URI")

class SectionExtractor(BaseTool):
    name: str = "section_extractor"
    description: str = "Extracts sections from a PDF using header detection."

    class InputSchema(BaseModel):
        file_path: str = Field(..., description="Absolute path to the PDF")

    args_schema = InputSchema

    def extract_sections(self, pages):
        chunks = []
        current_header = "Introduction"
        buffer = []

        for page_num, page in enumerate(pages, start=1):
            for line in page.split("\n"):
                if re.match(SECTION_REGEX, line.strip()):
                    if buffer:
                        chunks.append({
                            "header": current_header,
                            "content": "\n".join(buffer),
                            "page_number": page_num
                        })
                        buffer = []
                    current_header = line.strip()
                else:
                    buffer.append(line)

        if buffer:
            chunks.append({
                "header": current_header,
                "content": "\n".join(buffer),
                "page_number": len(pages)
            })

        return chunks

    def _run(self, file_path: str):
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        doc = fitz.open(file_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()

        sections = self.extract_sections(pages)
        return {
            "metadata": {"pages": len(pages), "file_path": file_path},
            "sections": sections
        }

load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEmbeddingGenerator:
    def __init__(self):
        self.model = GenerativeModel("text-embedding-004")

    def embed(self, text: str) -> list:
        response = self.model.embed_content(
            content=text,
        )
        return response["embedding"]

    def embed_sections(self, sections):
        for sec in sections:
            sec["embedding"] = self.embed(sec["content"])
        return sections


class MongoEmbeddingLoader:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.collection = self.client[db_name]["financial_chunks"]

    def clear_for_file(self, file_path: str):
        return self.collection.delete_many({"file_path": file_path}).deleted_count

    def upsert_sections(self, sections, metadata):
        ops = []
        file_path = metadata["file_path"]

        for sec in sections:
            ops.append(UpdateOne(
                {
                    "file_path": file_path,
                    "header": sec["header"],
                    "page_number": sec["page_number"],
                },
                {
                    "$set": {
                        "content": sec["content"],
                        "embedding": sec["embedding"],
                        "metadata": metadata
                    }
                },
                upsert=True,
            ))

        if ops:
            self.collection.bulk_write(ops)


class ProcessPDFTool(BaseTool):
    name: str = "process_pdf_tool"
    description: str = (
        "Extracts sections from a PDF, embeds them with Gemini, "
        "and stores embeddings in MongoDB."
    )

    class InputSchema(BaseModel):
        file_path: str = Field(..., description="Absolute PDF path")

    args_schema = InputSchema

    def _run(self, file_path: str):
        try:
            # Step 1: Extract sections
            extractor = SectionExtractor()
            result = extractor._run(file_path)

            if "error" in result:
                return result["error"]

            sections = result["sections"]
            metadata = result["metadata"]

            # Step 2: Generate embeddings
            embedder = GeminiEmbeddingGenerator()
            sections = embedder.embed_sections(sections)

            # Step 3: Store into MongoDB
            loader = MongoEmbeddingLoader(
                uri=os.getenv("MONGO_URI"),
                db_name="financial_docs"
            )

            loader.clear_for_file(file_path)
            loader.upsert_sections(sections, metadata)

            return (
                f"✅ Completed: {file_path}\n"
                f"Pages: {metadata['pages']}, Sections: {len(sections)}"
            )

        except Exception as e:
            return f"❌ Pipeline failed: {str(e)}"
