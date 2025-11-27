from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from crewai.tools.base_tool import BaseTool

from pydantic import BaseModel, Field

from typing import Dict, Any, List
import pymupdf as fitz
import os
from pathlib import Path


import re

import logging
import warnings


load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("camelot").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message="camelot.utils.TemporaryDirectory is deprecated.*")
warnings.filterwarnings("ignore", message="Cannot set gray non-stroke color because .*")

try:
    import camelot

    CAMELOT_INSTALLED = True
except ImportError:
    CAMELOT_INSTALLED = False

# Set up logging
logger = logging.getLogger(__name__)

SECTION_REGEX = r"(?P<header>^[A-Z][A-Za-z0-9 ,\-()]{3,}$)"
MONGO_URI = os.getenv("MONGO_URI")


class DocumentProcessing(BaseTool):
    """
    Combines PDF extraction + section chunking into a single tool.
    """

    name: str = "document_processing"
    description: str = (
        "Extracts text, metadata, tables, and structured sections from PDFs."
    )

    def __init__(self, **data):
        super().__init__(**data)

    # ------------ PDF Extraction ---------------

    def extract_text(self, doc):
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        return pages

    def extract_metadata(self, doc):
        metadata = doc.metadata
        return {
            'pages': doc.page_count,
            'author': metadata.get('author', 'Unknown'),
            'title': metadata.get('title', 'Unknown'),
            'subject': metadata.get('subject', 'Unknown'),
            'creator': metadata.get('creator', 'Unknown'),
            'producer': metadata.get('producer', 'Unknown'),
            'creation_date': metadata.get('creationDate', 'Unknown')
        }

    # ------------ Table Extraction -------------

    def extract_tables_from_pdf(self, pdf_path, doc):
        if not CAMELOT_INSTALLED:
            return []

        tables_all = []
        total_pages = doc.page_count

        for page in range(1, total_pages + 1):
            try:
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page),
                    flavor="lattice",
                    suppress_stdout=True
                )
                for table in tables:
                    if table.parsing_report["accuracy"] > 80:
                        tables_all.append({
                            "page": page,
                            "method": "camelot-lattice",
                            "data": table.df.to_dict("records")
                        })
            except Exception:
                continue

        return tables_all

    # ------------ Section Chunking -------------

    def chunk_pages(self, pages: List[str]):
        chunks = []
        current_header = "Introduction"
        buffer = []

        for page_num, page in enumerate(pages, start=1):
            lines = page.split("\n")
            for line in lines:
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
                "page_number": page_num
            })

        return chunks

    # ------------ Main Extraction Wrapper -------------

    def extract(self, file_path: str):
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        doc = None
        try:
            doc = fitz.open(file_path)

            pages = self.extract_text(doc)
            metadata = self.extract_metadata(doc)
            tables = self.extract_tables_from_pdf(file_path, doc)
            chunks = self.chunk_pages(pages)

            return {
                "metadata": metadata,
                "pages": pages,
                "tables": tables,
                "chunks": chunks
            }

        finally:
            if doc:
                doc.close()

    def _run(self, file_path: str):
        return self.extract(file_path)

    async def _arun(self, file_path: str):
        return self.extract(file_path)





class EmbeddingGenerator:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[Dict]):
        for chunk in chunks:
            chunk["embedding"] = self.model.encode(chunk["content"]).tolist()
        return chunks



class MongoLoader:

    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.collection = self.client[db_name]["financial_chunks"]

    def clear_document(self, title):
        """
        Remove all chunks for a specific document title.
        """
        delete_result = self.collection.delete_many({"metadata.title": title})
        return delete_result.deleted_count

    def upsert_chunks(self, chunks, metadata):
        ops = []

        for c in chunks:
            ops.append(UpdateOne(
                {
                    "header": c["header"],
                    "page_number": c["page_number"],
                    "title": metadata["title"]
                },
                {
                    "$set": {
                        "header": c["header"],
                        "page_number": c["page_number"],
                        "content": c["content"],
                        "embedding": c["embedding"],
                        "metadata": metadata
                    }
                },
                upsert=True
            ))

        if ops:
            self.collection.bulk_write(ops)


# pages = summary["text_content"]
#             metadata = summary["document_info"]

from pydantic import BaseModel, Field

class ProcessPDFToolInput(BaseModel):
    """Input schema for ProcessPDFTool."""
    file_path: str = Field(..., description="The **absolute file path** to the PDF document on the local file system.")


class ProcessPDFTool(BaseTool):
    """
    A tool to execute the full PDF processing pipeline: ingestion, chunking,
    embedding, and loading into the MongoDB vector store.
    """
    name: str = "process_pdf_tool"

    # 🎯 Tool Description for CrewAI Agent
    description: str = (
        "Executes the full pipeline to process a raw PDF file, extracting data and "
        "storing embedded chunks in MongoDB. **The single input must be the absolute file path** "
        "to the PDF. Use this tool first when analyzing a new financial report."
    )

    # 🎯 Pydantic Input Validation
    args_schema: type[BaseModel] = ProcessPDFToolInput
    def _run(self, file_path: str) -> str:
        """
        Input: Absolute file path to the PDF.
        Returns: Success message with document details or an error message.
        """
        if not os.path.exists(file_path):
            return f"Error: File not found at path: {file_path}"

        logger.info(f"--- Starting Pipeline for: {file_path} ---")
        try:
            # 1. Ingestion (PDFIngestionTool)
            ingestion = DocumentProcessing()
            summary = ingestion.extract(file_path)

            if "error" in summary:
                raise Exception(summary["error"])

            pages = summary["pages"]
            metadata = summary["metadata"]
            chunks = summary["chunks"]
            tables = summary["tables"]

            # # 2. Chunking (SectionChunker)
            # chunker = SectionChunker()
            # chunks = chunker.chunk_pages(pages)
            # logger.info(f"Structured sections extracted: {len(chunks)} chunks.")

            # 3. Embedding (EmbeddingGenerator - called within MongoLoader logic below)
            embedder = EmbeddingGenerator()
            embedded_chunks = embedder.embed_chunks(chunks)
            logger.info(f"Embeddings generated for all chunks.")

            # 4. Loading (MongoLoader)
            loader = MongoLoader(uri=MONGO_URI, db_name="financial_docs")
            # Clear old document embeddings FIRST
            title = metadata.get("title", os.path.basename(file_path))
            deleted = loader.clear_document(title)
            logger.info(f"Cleared {deleted} old chunks for title: {title}")
            loader.upsert_chunks(embedded_chunks, metadata)

            doc_title = metadata.get('title', os.path.basename(file_path))
            return (
                f"✅ Document Processing COMPLETE: '{doc_title}'\n"
                f"Pages: {metadata.get('pages', 'N/A')}, Chunks: {len(chunks)}, "
                f"Tables: {len(summary['tables'])}. Data is ready for query."
            )

        except Exception as e:
            logger.error(f"Pipeline failed for {file_path}: {e}")
            return f"Pipeline execution failed. Error: {str(e)}"

    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)


#
# class ProcessPDFTool(BaseTool):
#     """
#     A tool to execute the full PDF processing pipeline: ingestion, chunking,
#     embedding, and loading into the MongoDB vector store.
#     """
#     name: str = "process_pdf_tool"
#     description: str = (
#         "Executes the full pipeline to process a raw PDF file document to extract "
#         "metadata, text from each page, and high-confidence tables, chunking the text "
#         "into structured sections based on recognizable headers, generating sentence embeddings"
#         "for each chunk and storing chunks with metadata and embeddings as documents in MongoDB."
#     )
#
