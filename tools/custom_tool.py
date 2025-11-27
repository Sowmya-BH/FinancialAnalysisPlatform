import subprocess

from crewai.tools import BaseTool
from typing import Type

from pydantic import BaseModel, Field

import os
import json



class DoclingToolInput(BaseModel):
    """Input schema for DoclingTool."""
    pdf_file_name: str = Field(..., description="path of the PDF file")

class DoclingTool(BaseTool):
    name: str = "Docling"
    description: str = (
        "Tool used to parse input PDF file and convert it into JSON file"
    )
    args_schema: Type[BaseModel] = DoclingToolInput

    def _run(self, pdf_file_name: str) -> str:
        # Implementation goes here
        """Uses Docling tool to process a PDF file and convert it to a JSON file."""
        try:
            pdf_dir = os.path.dirname(pdf_file_name)
            cmd = f"docling --to json --force-ocr {os.path.basename(pdf_file_name)}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=pdf_dir)

            # # Running Docling as a command-line tool
            # cmd = f"docling --to json --force-ocr {pdf_file_name}"
            # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
            # result = 0
            # if result == 0:
                json_file = os.path.splitext(pdf_file_name)[0] + ".json"
                print("Document processed successfully! Output saved to file:", json_file)
                return json_file
            else:
                return f"Error processing document: {result.stderr}"
        except Exception as e:
            return f"Exception occurred: {str(e)}"



class Find_Next_Text_NodeInput(BaseModel):
    """Input schema for find_next_text_node."""
    file_name_json: str = Field(..., description="Name of the JSON file to parse")
    value: str = Field(..., description="The field to search for in the JSON file")


class Find_Next_Text_NodeInput(BaseModel):
    """Input schema for find_next_text_node."""
    file_name_json: str = Field(..., description="Name of the JSON file to parse")
    value: str = Field(..., description="The field to search for in the JSON file")


class Find_Next_Text_Node(BaseTool):
    name: str = "find_next_text_node"
    description: str = (
        "Recursively searches a JSON file and returns ALL 'text' values that appear AFTER a target value."
    )
    args_schema: Type[BaseModel] = Find_Next_Text_NodeInput

    def _run(self, file_name_json: str, value: str) -> str:
        print("JSON file:", file_name_json)

        with open(file_name_json, "r") as f:
            data = json.load(f)

        results = self.find_all_next_text_nodes(data, "text", value)

        if not results:
            return f"No values found after '{value}'."

        return {
            "keyword": value,
            "count": len(results),
            "values": results
        }

    # ---------------- FIXED FUNCTION -----------------
    def find_all_next_text_nodes(self, json_data, target_key, target_value):
        """
        Recursively scans JSON and returns ALL 'text' nodes AFTER a given key-value match.
        """

        results = []
        found_target = False

        def recursive_search(node):
            nonlocal found_target

            # Case 1: Node is a dictionary
            if isinstance(node, dict):
                for key, val in node.items():

                    # When we hit the target key/value
                    if key == target_key and val == target_value:
                        found_target = True

                    # Collect ALL "text" values after first occurrence
                    elif key == "text" and found_target:
                        results.append(val)

                    # Recurse deeper
                    if isinstance(val, (dict, list)):
                        recursive_search(val)

            # Case 2: Node is a list
            elif isinstance(node, list):
                for item in node:
                    recursive_search(item)

        recursive_search(json_data)
        return results
