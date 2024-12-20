from pipeline.pipeline_4 import run_pipeline_4

def main():
    # Define file paths
    file_path_excel = "2024-09-27_Food_Waren-und_Dienstleistungsgruppe_V_4.0.xlsx"
    file_paths_csv = "grouped_result.csv"
    result_path = "result/p4_matching_matrix.npy"
    embedding_cache_path = "embedding/Pipeline_2_embeddings_cache.pkl"

    # Set hyperparameters
    model_name = "gpt-4o-mini"  # Specify GPT model

    # Run Baseline Pipeline
    run_pipeline_4(file_path_excel, file_paths_csv, embedding_cache_path, result_path, gpt_model_name=model_name)


if __name__ == "__main__":
    main()