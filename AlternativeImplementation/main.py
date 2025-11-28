
import sys
import warnings

from datetime import datetime

# from crew import CkdV3

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


from crew import FullFinancialRAGCrew

def main():
    crew = FullFinancialRAGCrew().crew()

    results = crew.kickoff(
        inputs={
            "query": "What does the report say about total gross profit?"
        }
    )
    print("\n" + "=" * 50)
    print("✅ CREW EXECUTION FINISHED")
    print("=" * 50)

    # Print individual task results
    print(f"\n[Task 1: PDF Processing Status]\n{results['pdf_processing_status']}")
    print(f"\n[Task 2: RAG Search Result]\n{results['search_result']}")

    # The final output of the whole Crew is usually the result of the last task
    print("\nFINAL ANSWER:")
    print(results['search_result'])

    print("\n--- FINAL RESULT ---\n")
    print(results)

if __name__ == "__main__":
    main()
