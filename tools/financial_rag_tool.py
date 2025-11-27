# import os
# from crewai_tools import MongoDBVectorSearchConfig, MongoDBVectorSearchTool
# from crewai.tools import BaseTool
# from pymongo import MongoClient
#
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # 1. Configure the Search Parameters
# # This tells the tool how many chunks to retrieve (limit=20 is good for financial context)
# query_config = MongoDBVectorSearchConfig(
#     limit=20,
#     oversampling_factor=2  # Fetches 40, reranks/filters down to 10
# )
#
# # 2. Initialize the Tool
# # This tool connects to the data your ProcessPDFTool just created.
# financial_rag_tool = MongoDBVectorSearchTool(
#     connection_string=os.getenv("MONGO_URI"),
#     database_name="financial_docs",       # MUST match your MongoLoader
#     collection_name="financial_chunks",   # MUST match your MongoLoader
#     vector_index_name="vector_index",     # You must create this in Atlas UI
#     filter={},                            # Optional: Filter by {"metadata.title": "..."} if needed dynamically
#     query_config=query_config,
#     description=(
#         "Search the stored financial document embeddings. "
#         "Useful for finding specific values like 'Revenue', 'Net Income', "
#         "'Total Assets', or 'Liabilities' to calculate financial ratios."
#     )
# )

from crewai.tools import BaseTool
from pymongo import MongoClient
from pydantic import BaseModel, Field
import os
from sentence_transformers import SentenceTransformer


class MongoSearchInput(BaseModel):
    """Input for the search tool."""
    query: str = Field(..., description="The specific financial question or term to search for.")


class MongoSearchTool(BaseTool):
    name: str = "financial_db_search"
    description: str = "Search the financial database for relevant text and numbers."
    args_schema: type[BaseModel] = MongoSearchInput

    def _run(self, query: str) -> str:
        try:
            # 1. Embed the query
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_vector = model.encode(query).tolist()

            # 2. Connect
            client = MongoClient(os.getenv("MONGO_URI"))
            collection = client["financial_docs"]["financial_chunks"]

            # 3. Vector Search
            results = collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 50,
                        "limit": 10
                    }
                }
            ])

            # 4. Format Results
            output = ""
            for doc in results:
                output += f"Page {doc.get('page_number')}: {doc.get('content')}\n---\n"

            return output if output else "No relevant data found."

        except Exception as e:
            return f"Search failed: {str(e)}"