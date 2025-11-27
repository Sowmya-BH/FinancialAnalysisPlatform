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
        'user_query' : 'Your analysis of the performance of the company in a short paragraph with key highlights'
'

        # 'output_path':str(JSON_OUTPUT_PATH)
    }

    try:
        CkdV3().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    print("## Welcome to Stock Analysis Crew")
    main()
