import os
import pandas as pd

def load_and_process_data(excel_path, csv_path):
    """
    Load and process data from Excel and multiple CSV files.

    Args:
        excel_path (str): Path to the Excel file.
        csv_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - guidelines (list): List of guidelines from the Excel file.
            - sentences (list): List of sentences from the combined CSV files.
    """
    # Load Excel data
    data = pd.read_excel(excel_path, usecols=["Ausschreibungskriterium"])
    guidelines = data["Ausschreibungskriterium"].dropna().astype(str).tolist()

    # Load and combine CSV data
    data = pd.read_csv(csv_path, usecols=["sentence"])
    sentences = data["sentence"].dropna().astype(str).tolist()

    return guidelines, sentences


def ensure_directories_exist(directories):
    """
    Ensure the given directories exist, creating them if necessary.

    Args:
        directories (list): List of directory paths to ensure existence.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
