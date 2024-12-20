import os
import pandas as pd
import time
import difflib
import openai
import numpy as np
from pipeline.data_processing import load_and_process_data, ensure_directories_exist

import openai
import os

# Set the API key directly
openai.api_key = "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"

def gpt_baseline_search(guidelines, sentences, model_name="gpt-4o-mini"):
    M, N = len(sentences), len(guidelines)
    matching_matrix = np.zeros((M, N))

    # Iterate over each sentence and find top matches in guidelines
    for i, sentence in enumerate(sentences):
        print(f"Processing sentence: {sentence[:50]}...")  # Display progress
        messages = [
            {"role": "system", "content": "You are a helpful assistant trained for matching sentences with guidelines."},
            {"role": "user", "content": f"Sentence: {sentence}\nGuidelines: {guidelines}\nFind the most relevant guidelines to the sentence and rank them."}
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

        for idx, match in enumerate(matches):
            closest_match = difflib.get_close_matches(match, guidelines, n=1, cutoff=0.9)
            try:
                guideline_index = guidelines.index(closest_match[0])
                matching_matrix[i][guideline_index] = 1 - idx * 0.01
            except ValueError:
                print(f"nothing match: {match}.")

    return matching_matrix


def run_baseline(file_path_excel, file_paths_csv, result_path, model_name="gpt-4o-mini"):
    """
    Run the GPT baseline for semantic search.

    Args:
        file_path_excel (str): Path to the Excel file containing guidelines.
        file_paths_csv (list): List of CSV file paths containing requirements.
        result_path (str): Path to save results.
        model_name (str): The GPT model to use.

    Returns:
        None
    """
    # Ensure directories exist
    ensure_directories_exist(["result"])

    # Load and process data
    guidelines, sentences = load_and_process_data(file_path_excel, file_paths_csv)

    # Perform GPT-based semantic search
    print("Performing GPT-based semantic search...")
    start_time = time.time()
    matching_matrix = gpt_baseline_search(guidelines, sentences, model_name=model_name)
    elapsed_time = time.time() - start_time
    print(f"GPT-based semantic search completed in {elapsed_time:.2f} seconds.")

    # Save results to npy
    np.save(result_path, matching_matrix)
    print(f"Results saved to '{result_path}'.")
