from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool

# from tools.docling_tool import DoclingTool
# from tools.json_parse_tool import Find_Next_Text_Node
from tools.custom_tool import DoclingTool,Find_Next_Text_Node
from crewai_tools import FileWriterTool

from pydantic import BaseModel
from fastapi import FastAPI

from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
# ---------------------------------------------------------------------------
# LLM CONFIG
# ---------------------------------------------------------------------------
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.7
)

rom crewai_tools import MongoDBVectorSearchTool
from tools.process_pdf_tool import ProcessPDFTool  # your combined tool

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = BASE_DIR / "knowledge"
DOCUMENT_PATH = DOCUMENT_DIR / "TSLA-Q2-2025-Update.pdf"

# LLM configuration (uses env variable from YAML)
from crewai import LLM
llm = LLM(
    model=os.getenv("LLM_MODEL"),
    temperature=0.0
)


# ---------------------------------------------------------------------------
# CREW DEFINITION
# ---------------------------------------------------------------------------

@CrewBase
class FullFinancialRAGCrew():
    """
    RAG Crew: PDF ingestion → embedding → vector DB storage → querying.
    """

    agents_config = "config/full_rag_pipeline.yaml"
    tasks_config = "config/full_rag_pipeline.yaml"

    # --------------------------
    # 1. PDF PROCESSOR AGENT
    # --------------------------

    @agent
    def pdf_processor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["pdf_processor_agent"],
            llm=llm,
            verbose=True,
            tools=[
                ProcessPDFTool()
            ]
        )

    # --------------------------
    # 2. RAG SEARCH AGENT
    # --------------------------

    @agent
    def rag_search_agent(self) -> Agent:
        rag_tool = MongoDBVectorSearchTool(
            connection_string=os.getenv("MONGO_URI"),
            database_name="financial_docs",
            collection_name="financial_chunks",
            text_field="content",
            embedding_field="embedding",
            k=4
        )

        return Agent(
            config=self.agents_config["rag_search_agent"],
            llm=llm,
            verbose=True,
            tools=[rag_tool]
        )

    # --------------------------
    # TASKS
    # --------------------------

    @task
    def process_pdf_task(self):
        return {
            "config": self.tasks_config["process_pdf_task"],
            "input": {
                "DOCUMENT_PATH": str(DOCUMENT_PATH)
            }
        }

    @task
    def rag_search_task(self):
        return {
            "config": self.tasks_config["rag_search_task"]
            # Input "query" will be provided in kickoff(inputs={})
        }

    # --------------------------
    # CREW
    # --------------------------

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
