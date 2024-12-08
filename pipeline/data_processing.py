import os
import pandas as pd

def load_and_process_data(excel_path, csv_paths):
    """
    Load and process data from Excel and multiple CSV files.

    Args:
        excel_path (str): Path to the Excel file.
        csv_paths (list): List of paths to the CSV files.

    Returns:
        tuple: A tuple containing:
            - guidelines (list): List of combined texts (guidelines) from the Excel file.
            - requirements (list): List of sentences (requirements) from the combined CSV files.
    """
    # Load and process Excel data
    data = pd.read_excel(excel_path, usecols=["Kategorie Kriterium", "Ausschreibungskriterium"])
    guidelines = data.apply(lambda row: " ".join(row.dropna().astype(str)), axis=1).tolist()

    # Load and combine CSV data
    dataframes = [pd.read_csv(file_path) for file_path in csv_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    requirements = combined_df["text"].tolist()

    return guidelines, requirements


def ensure_directories_exist(directories):
    """
    Ensure the given directories exist, creating them if necessary.

    Args:
        directories (list): List of directory paths to ensure existence.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
