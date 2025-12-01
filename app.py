import os
import shutil
import json
import uuid
import datetime
from pathlib import Path
import functools
import time

from crewai import LLM
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,PlainTextResponse

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List


from crew import CkdV3  # your CrewAI class

gemini_api_key = os.getenv("GEMINI_API_KEY")

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.7
)

app = FastAPI(
    title="Financial Advisor API",
    description="AI-powered financial document analysis with CrewAI",
     # ReDoc
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
upload_dir = BASE_DIR / "knowledge"
# UPLOAD_DIR = BASE_DIR / "uploads"
# UPLOAD_DIR.mkdir(exist_ok=True)

# @app.get("/")
# async def root():
#     return {"message": "Welcome to Financial Advisor API"}

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "*Welcome to Financial Advisor API*"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single PDF file to the server.
    """

    # Ensure upload directory exists
    upload_dir = "./knowledge/uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": f"File {file.filename} is not a PDF"}
        )

    # Build final save path
    file_path = os.path.join(upload_dir, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Build response
    uploaded_file_info = {
        "filename": file.filename,
        "size": os.path.getsize(file_path),
        "path": file_path,
        "message": "Upload successful"
    }

    return {"uploadedFile": uploaded_file_info}


@app.post("/query")
async def handle_query(input: dict):
    try:
        query = input.get("query", "")
        result = CkdV3(query)

        result_str = str(result)

        return JSONResponse({
            "generateCopilotResponse": {
                "response": result_str
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/get-pdf-details/{filename}")
async def get_pdf_details(filename: str):
    """
    Retrieve the filename and size for a single, specific PDF file
    in the 'knowledge/uploaded_pdfs' directory.
    """
    try:
        pdf_directory = "uploaded_pdfs"

        # Construct the full expected file path
        full_directory = os.path.join("knowledge", pdf_directory)
        file_path = os.path.join(full_directory, filename)

        # 1. Check if the directory exists
        if not os.path.exists(full_directory):
            return JSONResponse(
                status_code=404,
                content={"error": f"The PDF storage directory '{full_directory}' was not found."}
            )

        # 2. Check if the file exists and is a file
        if not os.path.isfile(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"File '{filename}' not found in '{full_directory}'."}
            )

        # 3. Check if the file has a .pdf extension (optional safety check)
        if not filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"error": "The requested file is not a PDF."}
            )

        # 4. If all checks pass, get the details
        file_details = {
            "filename": filename,
            "size": os.path.getsize(file_path)
        }

        return {"file": file_details}

    except Exception as e:
        # Catch any unexpected errors (e.g., permission issues)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving PDF details: {str(e)}"}
        )


# Retrieve details for a single specific PDF file ---
@app.get("/get-pdf-details/{filename}")
async def get_pdf_details(filename: str):
    """
    Retrieve the filename and size for a single, specific PDF file
    in the 'knowledge/uploaded_pdfs' directory.
    """
    try:
        pdf_directory = "uploaded_pdfs"

        # Construct the full expected file path
        full_directory = os.path.join("knowledge", pdf_directory)
        file_path = os.path.join(full_directory, filename)

        # 1. Check if the directory exists
        if not os.path.exists(full_directory):
            return JSONResponse(
                status_code=404,
                content={"error": f"The PDF storage directory '{full_directory}' was not found."}
            )

        # 2. Check if the file exists and is a file
        if not os.path.isfile(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"File '{filename}' not found in '{full_directory}'."}
            )

        # 3. Check if the file has a .pdf extension (optional safety check)
        if not filename.lower().endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"error": "The requested file is not a PDF."}
            )

        # 4. If all checks pass, get the details
        file_details = {
            "filename": filename,
            "size": os.path.getsize(file_path)
        }

        return {"file": file_details}

    except Exception as e:
        # Catch any unexpected errors (e.g., permission issues)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error retrieving PDF details: {str(e)}"}
        )

message_cache = {}
agent_states = {}

@app.post("/copilot-compatible")
async def handle_copilot_query(request: Request):
    try:
        body = await request.json()
        print("Received request body:", body)

        operation_name = body.get("operationName", "")

        if operation_name == "availableAgents":
            return JSONResponse({
                "data": {
                    "availableAgents": {
                        "agents": [
                            {
                                "name": "MyCopilotCrew",
                                "id": "MyCopilotCrew",
                                "description": "A CrewAI agent that can analyze PDF documents"
                            }
                        ]
                    }
                }
            })

        elif operation_name == "loadAgentState":
            variables = body.get("variables", {})
            data = variables.get("data", {})
            agent_name = data.get("agentName", "")
            thread_id = data.get("threadId", "")

            if thread_id not in agent_states:
                agent_states[thread_id] = {
                    "inputs": {},
                    "result": "Crew result will appear here..."
                }

            return JSONResponse({
                "data": {
                    "loadAgentState": {
                        "threadId": thread_id,
                        "threadExists": True,
                        "state": json.dumps(agent_states[thread_id]),
                        "messages": []
                    }
                }
            })

        elif operation_name == "generateCopilotResponse":
            variables = body.get("variables", {})
            data = variables.get("data", {})
            thread_id = data.get("threadId", "")
            messages = data.get("messages", [])
            agent_states_data = data.get("agentStates", [])

            user_message = None
            parent_id = None
            for msg in messages:
                if "textMessage" in msg:
                    text_message = msg.get("textMessage", {})
                    if text_message.get("role") == "user":
                        user_message = text_message.get("content", "")
                        parent_id = msg.get("id")
                        break

            if not user_message:
                for msg in messages:
                    if "textMessage" in msg:
                        text_message = msg.get("textMessage", {})
                        if text_message.get("role") in ["developer", "assistant"]:
                            user_message = text_message.get("content", "")
                            parent_id = msg.get("id")
                            break

            if not user_message or "provide your inputs" in user_message.lower():
                user_message = "Summarize the PDF documents"

            cache_key = user_message

            current_time = time.time()
            if cache_key in message_cache:
                cached_time, cached_result = message_cache[cache_key]
                if current_time - cached_time < 30:
                    print(f"Using cached result for: {user_message}")
                    crew_result = cached_result
                else:
                    crew_result = CkdV3(user_message)
                    message_cache[cache_key] = (current_time, crew_result)
            else:
                crew_result = CkdV3(user_message)
                message_cache[cache_key] = (current_time, crew_result)

            agent_states[thread_id] = {
                "inputs": {"query": user_message},
                "result": str(crew_result)
            }

            message_id = f"msg_{uuid.uuid4()}"

            return JSONResponse({
                "data": {
                    "generateCopilotResponse": {
                        "threadId": thread_id,
                        "runId": f"run_{uuid.uuid4()}",
                        "messages": [
                            {
                                "__typename": "TextMessageOutput",
                                "id": message_id,
                                "createdAt": datetime.datetime.now().isoformat(),
                                "content": [str(crew_result)],
                                "role": "assistant",
                                "parentMessageId": parent_id
                            },
                            {
                                "__typename": "AgentStateMessageOutput",
                                "id": f"state_{uuid.uuid4()}",
                                "createdAt": datetime.datetime.now().isoformat(),
                                "threadId": thread_id,
                                "state": json.dumps(agent_states[thread_id]),
                                "running": False,
                                "agentName": "MyCopilotCrew",
                                "nodeName": "MyCopilotCrew",
                                "runId": f"run_{uuid.uuid4()}",
                                "active": True,
                                "role": "assistant"
                            }
                        ]
                    }
                }
            })

        return JSONResponse({
            "errors": [{"message": f"Unsupported operation: {operation_name}"}]
        }, status_code=400)

    except Exception as e:
        print(f"Error in copilot-compatible endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "errors": [{"message": str(e)}]
        }, status_code=500)


@app.post("/debug-request")
async def debug_request(request: Request):
    try:
        body = await request.json()
        print("Debug endpoint received:", body)
        return JSONResponse({
            "received": body,
            "message": "Request received successfully"
        })
    except Exception as e:
        print(f"Error in debug endpoint: {str(e)}")
        return JSONResponse({
            "error": str(e)
        }, status_code=500)
