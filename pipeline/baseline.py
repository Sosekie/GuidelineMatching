import os
import pandas as pd
import time
import openai
from pipeline.data_processing import load_and_process_data, ensure_directories_exist

import openai
import os

# Set the API key directly
openai.api_key = "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"

def gpt_baseline_search(guidelines, requirements, model_name="gpt-4"):
    """
    Perform semantic search using GPT as the baseline.

    Args:
        guidelines (list): List of guideline texts (queries).
        requirements (list): List of requirement texts (corpus).
        model_name (str): The GPT model to use for semantic search.

    Returns:
        list: List of dictionaries containing query and matched results.
    """
    results = []

    # Iterate over each guideline and find top matches in requirements
    for guideline in guidelines:
        print(f"Processing guideline: {guideline[:50]}...")  # Display progress
        messages = [
            {"role": "system", "content": "You are a helpful assistant trained for matching guidelines with requirements."},
            {"role": "user", "content": f"Guideline: {guideline}\nRequirements: {requirements}\nFind the most relevant requirements to the guideline and rank them."}
        ]

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0  # Deterministic output
        )
        # Parse GPT output
        output = response["choices"][0]["message"]["content"]

        # Extract matches
        try:
            matches = [{"text": line.strip()} for line in output.split("\n") if line.strip()]
        except Exception as e:
            print(f"Error parsing GPT output: {e}")
            matches = []

        results.append({"query": guideline, "matches": matches})

    return results


def run_baseline(file_path_excel, file_paths_csv, result_path, model_name="gpt-4", top_k=10):
    """
    Run the GPT baseline for semantic search.

    Args:
        file_path_excel (str): Path to the Excel file containing guidelines.
        file_paths_csv (list): List of CSV file paths containing requirements.
        result_path (str): Path to save results.
        model_name (str): The GPT model to use.
        top_k (int): Number of top matches to retrieve.

    Returns:
        None
    """
    # Ensure directories exist
    ensure_directories_exist(["result"])

    # Load and process data
    guidelines, requirements = load_and_process_data(file_path_excel, file_paths_csv)

    # Perform GPT-based semantic search
    print("Performing GPT-based semantic search...")
    start_time = time.time()
    gpt_results = gpt_baseline_search(guidelines, requirements, model_name=model_name)
    elapsed_time = time.time() - start_time
    print(f"GPT-based semantic search completed in {elapsed_time:.2f} seconds.")

    # Save results to CSV
    results_list = []
    for result in gpt_results:
        query = result["query"]
        for idx, match in enumerate(result["matches"][:top_k]):
            results_list.append({"query": query, "matched_text": match["text"], "rank": idx + 1})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(result_path, index=False)
    print(f"Results saved to '{result_path}'.")
