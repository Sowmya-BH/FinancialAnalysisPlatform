#!/usr/bin/env python

# from crew import FinancialAdvisor, DOCUMENT_PATH, DOCUMENT_DIR
# from financial_advisor.crew import FinancialAdvisor
from pathlib import Path
import os

import sys
import warnings

from datetime import datetime

from crew import CkdV3

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def main():
    print("🚀 Starting Financial Document Analysis...\n")
    BASE_DIR = Path(__file__).resolve().parent
    DOCUMENT_DIR = BASE_DIR / "knowledge"
    JSON_OUTPUT_PATH = BASE_DIR.parent / "TSLA-Q2-2025-Update.json"


    DOCUMENT_PATH = DOCUMENT_DIR / "TSLA-Q2-2025-Update.pdf"
    print("PDF path:", DOCUMENT_PATH)
    print(os.path.basename)
    print("output_p",os.path.basename("TSLA-Q2-2025-Update.pdf"))
    print("Exists?", DOCUMENT_PATH.exists())
    """
    Run the crew.
    """

    inputs = {
        'pdf_path': str(DOCUMENT_PATH),
        'input_field': 'Total gross profit',
        'user_query' : 'Analysis on Total gross profit '

        # 'output_path':str(JSON_OUTPUT_PATH)
    }

    try:
        CkdV3().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    print("## Welcome to Stock Analysis Crew")
    main()
#
#
# # This main file is intended to be a way for you to run your
# # crew locally, so refrain from adding unnecessary logic into this file.
# # Replace with inputs you want to test with, it will automatically
# # interpolate any tasks and agents information
#
# def main():
#     """
#     Run the crew.
#     """
#     print("🚀 Starting Financial Document Analysis...\n")
#     BASE_DIR = Path(__file__).resolve().parent
#     DOCUMENT_DIR = BASE_DIR.parent.parent / "knowledge"
#
#     DOCUMENT_PATH = DOCUMENT_DIR / "TSLA-Q2-2025-Update.pdf"
#     print("PDF path:", DOCUMENT_PATH)
#     print("Exists?", DOCUMENT_PATH.exists())
#     doc_path = '/Users/bhupatiraju/PycharmProjects/financial_Report_Analyst/financial_advisor/knowledge/TSLA-Q2-2025-Update.pdf'
#
#     inputs = {
#         'pdf_path': str(DOCUMENT_PATH),
#         "question":"What is the paper about? what are its highlights."
#     }
#
#     try:
#         FinancialAdvisor().crew().kickoff(inputs=inputs)
#
#     except Exception as e:
#         raise Exception(f"An error occurred while running the crew: {e}")
#
#
# if __name__ == "__main__":
#     print("## Welcome to Stock Analysis Crew")
#     main()
    # print("\n\n########################")
    # print("## Here is the Report")
    # print("########################\n")
# def train():
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {
#         "topic": "AI LLMs",
#         'current_year': str(datetime.now().year)
#     }
#     try:
#         FinancialAdvisor().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
#
#     except Exception as e:
#         raise Exception(f"An error occurred while training the crew: {e}")
#
# def replay():
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         FinancialAdvisor().crew().replay(task_id=sys.argv[1])
#
#     except Exception as e:
#         raise Exception(f"An error occurred while replaying the crew: {e}")
#
# def test():
#     """
#     Test the crew execution and returns the results.
#     """
#     inputs = {
#         "topic": "AI LLMs",
#         "current_year": str(datetime.now().year)
#     }
#
#     try:
#         FinancialAdvisor().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
#
#     except Exception as e:
#         raise Exception(f"An error occurred while testing the crew: {e}")
#
# def run_with_trigger():
#     """
#     Run the crew with trigger payload.
#     """
#     import json
#
#     if len(sys.argv) < 2:
#         raise Exception("No trigger payload provided. Please provide JSON payload as argument.")
#
#     try:
#         trigger_payload = json.loads(sys.argv[1])
#     except json.JSONDecodeError:
#         raise Exception("Invalid JSON payload provided as argument")
#
#     inputs = {
#         "crewai_trigger_payload": trigger_payload,
#         "topic": "",
#         "current_year": ""
#     }
#
#     try:
#         result = FinancialAdvisor().crew().kickoff(inputs=inputs)
#         return result
#     except Exception as e:
#         raise Exception(f"An error occurred while running the crew with trigger: {e}")
# User name is John Doe.
# User is an AI Engineer.
# User is interested in AI Agents.
# User is based in San Francisco, California.
