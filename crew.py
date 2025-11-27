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

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
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

    # @agent
    # def file_writer_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['file_writer_agent'],
    #         verbose=True,
    #         tools=[FileWriterTool()],
    #         output_file="parsed_financial_document.json",
    #         llm=llm,
    #     )


    @agent
    def JSON_data_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['JSON_data_extractor'],
            verbose=True,
            tools=[DoclingTool(),
                   Find_Next_Text_Node()],
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
            context=[self.parse_pdf()],
            output_file = 'output.md',
        )
    # @task
    # def parse_pdf(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['parse_pdf'],
    #         input={
    #             "pdf_path": str(DOCUMENT_PATH),
    #         },
    #         output_file='output.md',
    #     )
    #
    # @task
    # def parse_json(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['parse_json'],
    #         context=[self.parse_pdf()],
    #         # NEW: Explicitly providing both required paths to the task
    #         # input={"json_file_path": str(JSON_OUTPUT_PATH)},
    #         output_file="financial_health_report.md",
    #
    #     )

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
# # ---------------------------------------------------------------------------
# # CREW DEFINITION
# # ---------------------------------------------------------------------------
# @CrewBase
# class FinancialAdvisor:
#     """FinancialAdvisor Crew"""
#
#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"
#
#     # ------------------- AGENTS -------------------
#     @agent
#     def financial_analyst(self) -> Agent:
#         return Agent(
#             config=self.agents_config["financial_analyst"],
#             verbose=True,
#             llm=llm,
#             tools=[pdf_search_tool],
#         )
#
#     # ------------------- TASKS -------------------
#     @task
#     def answer_question_from_pdf(self) -> Task:
#         return Task(
#             config=self.tasks_config["answer_question_from_pdf"],
#             agent=self.financial_analyst(),
#             output_file="financial_health_report.md",
#             verbose=True,
#         )
#
#     # ------------------- CREW -------------------
#     @crew
#     def crew(self) -> Crew:
#         return Crew(
#             agents=[self.financial_analyst()],
#             tasks=[self.answer_question_from_pdf()],
#             process=Process.sequential,
#             verbose=True,
#         )
#

# from crewai import Agent, Crew, Process, Task
# from crewai.project import CrewBase, agent, crew, task
# from crewai.agents.agent_builder.base_agent import BaseAgent
# # from crewai_tools import process_pdf_tool
# # from tools.calculator_tool import CalculatorTool
# # import os
# # from tools.financial_rag_tool import MongoSearchTool
# # from tools.process_pdf_tool import ProcessPDFTool
# from crewai_tools import DirectoryReadTool,FileReadTool,PDFSearchTool
# from sentence_transformers import SentenceTransformer
# from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
# # from langchain_core.language_models.chat_models import LangSmithParams
#
# from crewai import LLM
# from openai import OpenAI
#
# import os
# from pathlib import Path
# from dotenv import load_dotenv
#
# load_dotenv()
# api_key=os.getenv("GEMINI_API_KEY")
# openai_api_key=os.getenv("OPENAI_API_KEY")
#
# llm = LLM(
#     model="gemini/gemini-2.5-flash",
#     api_key= api_key,  # Or set GOOGLE_API_KEY/GEMINI_API_KEY
#     temperature=0.7
# )
#
# # Create a knowledge source
# # content_source = CrewDoclingSource(
# #     file_paths=[
# #         'TSLA-Q2-2025-Update.pdf',
# #     ],
# # )
# # model_name = 'all-MiniLM-L6-v2'
# # embedding_model = SentenceTransformer(model_name)
# #
# # # Define a simple wrapper to match expected interface of pdftool
# # class SentenceTransformerWrapper:
# #     def __init__(self, model):
# #         self.model = model
# #
# #     def embed(self, texts):
# #         if isinstance(texts, str):
# #             texts = [texts]
# #         return self.model.encode(texts).tolist()  # convert numpy to list if needed
# #
# # # Wrap the model
# # wrapped_embedding_model = SentenceTransformerWrapper(embedding_model)
# #
# #
# #
# # # --- Define the EMBEDDINGS configuration as a dictionary (THE FIX) ---
# # # The tool's RAG backend needs to know how to initialize the SentenceTransformer
# # embedding_tool_config = {
# #     "provider": "huggingface",
# #     "config": {
# #         "model": "all-MiniLM-L6-v2"
# #     }
# # }
# #
# # # --- Define the LLM configuration as a dictionary for the pdf tool ---
# # llm_tool_config = {
# #     "provider": "google",
# #     "config": {
# #         "model": "gemini-2.5-flash",
# #         "api_key": api_key,
# #         "temperature": 0.7
# #     }
# # }
# #
# # ---------------------------------------------------------------------------
# # PATH CONFIGURATION
# # ---------------------------------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent
# DOCUMENT_DIR = BASE_DIR.parent.parent / "knowledge"
# DOCUMENT_PATH = DOCUMENT_DIR / "TSLA-Q2-2025-Update.pdf"
#
# # # FINAL TOOL INITIALIZATION
# # pdf_tool = PDFSearchTool(
# #     pdf_path=DOCUMENT_PATH,
# #     config={
# #         "llm": llm_tool_config,
# #         # FIX: Pass the dictionary configuration, not the wrapped object
# #         "embedder": embedding_tool_config
# #     }
# # )
# pdf_search_tool = PDFSearchTool(
#     pdf=str(DOCUMENT_PATH),
#     config=dict(
#         llm=dict(provider="google", config=dict(model="gemini-2.5-flash",api_key=api_key)),
#         embedder=dict(
#             provider="huggingface",
#             config=dict(
#                 model="BAAI/bge-small-en-v1.5" # <-- NEW MODEL
#             )
#         ),
#     ),
# )
#
# # from typing import List
# #
# # # If you want to run a snippet of code before or after the crew starts,
# # # you can use the @before_kickoff and @after_kickoff decorators
# # # https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
# #
# # # ---------------------------------------------------------------------------
# # # PATH CONFIGURATION
# # # ---------------------------------------------------------------------------
# # BASE_DIR = Path(__file__).resolve().parent
# # DOCUMENT_DIR = BASE_DIR.parent.parent / "knowledge"
# #
# # DOCUMENT_PATH = DOCUMENT_DIR / "TSLA-Q2-2025-Update.pdf"
# #
# # pdf_tool = PDFSearchTool(
# #     pdf_path=DOCUMENT_PATH,
# #     config={
# #         "llm": llm_tool_config,
# #         "embeddings": wrapped_embedding_model
# #     }
# # )
# # #
# # # --- RAG TOOL CONFIGURATION INSIDE CREW ---
# # query_config = MongoDBVectorSearchConfig(limit=15, oversampling_factor=2)
# # financial_rag_tool = MongoDBVectorSearchTool(
# #     connection_string=os.getenv("MONGO_URI"),
# #     database_name="financial_docs",
# #     collection_name="financial_chunks",
# #     vector_index_name="vector_index",
# #     query_config=query_config
# # )
#
#
#
# # Knowledge sources
#
# # ---------------------------------------------------------------------------
# # CREW DEFINITION
# # ---------------------------------------------------------------------------
# @CrewBase
# class FinancialAdvisor():
#     """FinancialAdvisor crew"""
#     agents_config = 'config/agents.yaml'
#     tasks_config = 'config/tasks.yaml'
#
#     # -----------------------------------------------------------------------
#     # Agents
#     # -----------------------------------------------------------------------
#     @agent
#     def financial_analyst(self) -> Agent:
#         """Agent responsible for document ingestion and analysis."""
#         return Agent(
#             config=self.agents_config["financial_analyst"],
#             verbose=True,
#             llm=llm,
#             tools=[pdf_search_tool]
#         )
#
#     # @agent
#     # def investment_advisor_agent(self) -> Agent:
#     #     """Agent responsible for retrieving data and calculating ratios."""
#     #     # If you haven't added 'financial_ratio_calculator' to agents.yaml yet,
#     #     # you can reuse 'financial_analyst' config temporarily, but separate is better.
#     #     return Agent(
#     #         config=self.agents_config["investment_advisor_agent"],
#     #         verbose=True,
#     #         llm=llm,
#     #         tools=[
#     #             # This tool pulls data FROM MongoDB using your working code
#     #             MongoSearchTool(),
#     #             CalculatorTool()
#     #         ],
#     #     )
#
#     # -----------------------------------------------------------------------
#     # Tasks
#     # -----------------------------------------------------------------------
#     @task
#     def answer_question_from_pdf(self) -> Task:
#         return Task(
#             config=self.tasks_config["answer_question_from_pdf"],
#             agent=self.financial_analyst(),
#             output_file='financial_health_report.md',
#             verbose=True,
#         )
#
#     # @task
#     # def calculate_ratios_and_assess_health_task(self) -> Task:
#     #     """
#     #     Task 2: Retrieve data, calculate ratios, and generate the final report.
#     #     """
#     #     return Task(
#     #         config=self.tasks_config["calculate_ratios_and_assess_health_task"],
#     #         agent=self.investment_advisor_agent(),
#     #         output_file='financial_health_report.md',  # Saves the final output here
#     #         verbose=True,
#     #         # Explicitly state context to ensure sequential execution logic
#     #         context=[self.analyze_pdf_task()]
#     #     )
#     # -----------------------------------------------------------------------
#     # Crew Assembly
#     # -----------------------------------------------------------------------
#     @crew
#     def crew(self) -> Crew:
#         """Creates the Financial Document Analysis Crew."""
#         return Crew(
#             # Order matters! Analyst loads data first, Advisor reads it second.
#             agents=[
#                 self.financial_analyst()
#                 # self.investment_advisor_agent()
#             ],
#             tasks=[
#                 self.answer_question_from_pdf()
#                 # self.calculate_ratios_and_assess_health_task()
#             ],
#             process=Process.sequential,
#             verbose=True,
#             # knowledge_sources=[
#             #     content_source
#             # ]
#         )
#         # return Crew(
#         #     agents=self.agents,
#         #     tasks=self.tasks,
#         #     process=Process.sequential,
#         #     verbose=True,
#         # )
#
#
#     #
#     # agents: List[BaseAgent]
#     # tasks: List[Task]
#     #
#     # # Learn more about YAML configuration files here:
#     # # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
#     # # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
#     #
#     # # If you would like to add tools to your agents, you can learn more about it here:
#     # # https://docs.crewai.com/concepts/agents#agent-tools
#     # @agent
#     # def researcher(self) -> Agent:
#     #     return Agent(
#     #         config=self.agents_config['researcher'], # type: ignore[index]
#     #         verbose=True
#     #     )
#     #
#     # @agent
#     # def reporting_analyst(self) -> Agent:
#     #     return Agent(
#     #         config=self.agents_config['reporting_analyst'], # type: ignore[index]
#     #         verbose=True
#     #     )
#     #
#     # # To learn more about structured task outputs,
#     # # task dependencies, and task callbacks, check out the documentation:
#     # # https://docs.crewai.com/concepts/tasks#overview-of-a-task
#     # @task
#     # def research_task(self) -> Task:
#     #     return Task(
#     #         config=self.tasks_config['research_task'], # type: ignore[index]
#     #     )
#     #
#     # @task
#     # def reporting_task(self) -> Task:
#     #     return Task(
#     #         config=self.tasks_config['reporting_task'], # type: ignore[index]
#     #         output_file='report.md'
#     #     )
#     #
#     # @crew
#     # def crew(self) -> Crew:
#     #     """Creates the FinancialAdvisor crew"""
#     #     # To learn how to add knowledge sources to your crew, check out the documentation:
#     #     # https://docs.crewai.com/concepts/knowledge#what-is-knowledge
#     #
#     #     return Crew(
#     #         agents=self.agents, # Automatically created by the @agent decorator
#     #         tasks=self.tasks, # Automatically created by the @task decorator
#     #         process=Process.sequential,
#     #         verbose=True,
#     #         # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
#     #     )
