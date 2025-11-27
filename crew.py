from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import PDFSearchTool

# from tools.docling_tool import DoclingTool
# from tools.json_parse_tool import Find_Next_Text_Node
from tools.custom_tool import DoclingTool,Find_Next_Text_Node
from crewai_tools import FileWriterTool


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
# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = BASE_DIR.parent.parent / "knowledge"
DOCUMENT_PATH = DOCUMENT_DIR / "TSLA-Q2-2025-Update.pdf"
JSON_OUTPUT_PATH = BASE_DIR.parent / "TSLA-Q2-2025-Update.json"


# # ---------------------------------------------------------------------------
# # CREW DEFINITION
# # ---------------------------------------------------------------------------
@CrewBase
class CkdV3():
    """CkdV3 crew"""

    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def docling_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['docling_agent'],
            verbose=True,
            tools=[DoclingTool(),
                   FileWriterTool()],
            llm=llm,
        )


    @agent
    def JSON_data_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['JSON_data_extractor'],
            verbose=True,
            tools=[DoclingTool(),
                   Find_Next_Text_Node()],
            llm=llm,
        )

     @agent
    def financial_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst_agent'],
            verbose=True,
            tools=[],
            llm=llm,
        )
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def parse_pdf(self) -> Task:
        return Task(
            config=self.tasks_config['parse_pdf'],
        )

    @task
    def parse_json(self) -> Task:
        return Task(
            config=self.tasks_config['parse_json'],
            context=[self.parse_pdf()], #  Use output of 'parse_pdf' (the JSON path) as context for this task.
            output_file = 'output.md',
        )

    # Consumes the JSON output path and answers the user's query.
    @task
    def answer_query(self) -> Task:
        return Task(
            config=self.tasks_config['answer_query'],
            context=[self.parse_pdf()], #  Use output of 'parse_pdf' (the JSON path) as context for this task.
            output_file='summary.md',
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the CkdV3 crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
# # ---------------------------------------------------------------------------
# # PDF SEARCH TOOL (RAG)
# # ---------------------------------------------------------------------------
# # Assuming gemini_api_key is defined and DOCUMENT_PATH is correct
#
# pdf_search_tool = PDFSearchTool(
#     pdf=str(DOCUMENT_PATH),
#     config=dict(
#         llm=dict(
#             provider="gemini",
#             config=dict(
#                 model="gemini-2.5-flash",
#                 api_key=gemini_api_key,
#             )
#         ),
#         vectordb=dict(
#             provider="chromadb",
#             config=dict(
#                 persist_directory="vector_store",
#             ),
#         ),
#         # FIX: Change 'embedder' to the internal key 'embedding_model'
#         embedder=dict(
#             provider="google",
#             config=dict(
#                 model="models/embedding-001",
#                 task_type="retrieval_document",
#                 # title="Embeddings",
#             ),
#         ),
#     )
# )
