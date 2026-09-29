"""
ASTRA INTEL - Retrieval Engine Evaluator
=========================================

What does this script do?
-------------------------
In RAG systems, the quality of the answer depends entirely on finding the right text chunks.
If the search engine fails to retrieve the correct page, the LLM cannot answer correctly.

This script measures the "Accuracy" of our local vector search engine.
You provide a PDF and a JSON file containing sample questions and the correct page number
where the answer to that question can be found.

Metrics Explained:
- Top-1 Accuracy: Did the search engine return the correct page as its #1 absolute best match?
- Top-K Accuracy: Did the search engine return the correct page somewhere in its Top K matches?

How to Run:
-----------
    python tests/eval.py path/to/document.pdf tests/eval_questions.example.json
"""

import json
import sys
import os
from typing import Any, Dict, List

# Add the 'src' directory to the python path so we can import our RAG engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# pyrefly: ignore [missing-import]
from rag import DocIndex, TOP_K, DocError

def main() -> None:
    # 1. Parse command line arguments
    try:
        pdf_path = sys.argv[1]
        questions_path = sys.argv[2]
    except IndexError:
        print("Usage: python tests/eval.py <path_to_pdf> <path_to_questions.json>")
        sys.exit(1)

    # 2. Load the PDF document securely into the memory index using context managers
    print(f"Loading document: {pdf_path}...")
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        index = DocIndex(pdf_bytes, pdf_path)
    except FileNotFoundError:
        print(f"Error: The PDF file '{pdf_path}' was not found.")
        sys.exit(1)
    except DocError as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)

    # 3. Load the evaluation questions dataset
    print(f"Loading questions dataset: {questions_path}...\n")
    try:
        with open(questions_path, "r", encoding="utf-8") as q_file:
            cases: List[Dict[str, Any]] = json.load(q_file)
    except FileNotFoundError:
        print(f"Error: The JSON questions file '{questions_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{questions_path}' is not a valid JSON file.")
        sys.exit(1)

    total_cases = len(cases)
    if total_cases == 0:
        print("The questions JSON file is empty. Nothing to evaluate.")
        sys.exit(0)

    # Track how many times the system correctly found the right page
    hit_top1 = 0
    hit_topk = 0
    actual_k = 0

    print("--- Evaluation Results ---")
    for case in cases:
        question = case.get("q", "")
        expected_page = case.get("page", 0)

        # Run the search query through the vector index
        results = index.search(question)
        
        # Extract the page numbers from the retrieved chunks
        retrieved_pages = [r["page"] for r in results]
        
        if len(retrieved_pages) > 0:
            actual_k = len(retrieved_pages)  # Typically equal to TOP_K unless document is tiny
            is_top1_correct = (retrieved_pages[0] == expected_page)
            is_topk_correct = (expected_page in retrieved_pages)
        else:
            is_top1_correct = False
            is_topk_correct = False

        hit_top1 += is_top1_correct
        hit_topk += is_topk_correct
        
        # Display the result for this question
        status = "✅ OK " if is_topk_correct else "❌ MISS"
        print(f"{status} | Expected p.{expected_page} <- Retrieved {retrieved_pages} | {question}")

    # 4. Print final metrics
    print("\n--- Final Metrics ---")
    print(f"Total Questions Evaluated: {total_cases}")
    print(f"Top-1 Page Accuracy: {hit_top1}/{total_cases} = {hit_top1 / total_cases:.0%}")
    print(f"Top-{max(actual_k, TOP_K)} Page Accuracy: {hit_topk}/{total_cases} = {hit_topk / total_cases:.0%}")

if __name__ == "__main__":
    main()
