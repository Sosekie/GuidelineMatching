import os
import pickle
import pandas as pd
from time import time
from sentence_transformers import SentenceTransformer, util
from pipeline.data_processing import load_and_process_data, ensure_directories_exist


def semantic_search_global(input_queries, corpus_embeddings, corpus_texts, model, top_k=5):
    """
    Perform global semantic search for each input query.

    Args:
        input_queries (list): List of input queries (guidelines).
        corpus_embeddings (tensor): Precomputed embeddings of the corpus (requirements).
        corpus_texts (list): Corresponding texts of the corpus embeddings (requirements).
        model (SentenceTransformer): SentenceTransformer model for encoding.
        top_k (int): Number of top matches to return.

    Returns:
        list: List of dictionaries containing query and matched results.
    """
    # Encode the input queries
    query_embeddings = model.encode(input_queries, convert_to_tensor=True, show_progress_bar=True)

    # Perform semantic search
    all_results = []
    for query_idx, query_embedding in enumerate(query_embeddings):
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        # Collect results for this query
        query_result = {
            "query": input_queries[query_idx],
            "matches": [{"score": float(hit["score"]), "text": corpus_texts[hit["corpus_id"]]} for hit in hits]
        }
        all_results.append(query_result)

    return all_results


def generate_embeddings(model, requirements, guidelines, embedding_cache_path, batch_size=32):
    """
    Generate or load embeddings for requirements and guidelines.

    Args:
        model (SentenceTransformer): The embedding model.
        requirements (list): List of requirement sentences.
        guidelines (list): List of guideline texts.
        embedding_cache_path (str): Path to cache embeddings.
        batch_size (int): Batch size for embedding generation.

    Returns:
        tuple: Tuple containing requirement embeddings and guideline embeddings.
    """
    if not os.path.exists(embedding_cache_path):
        print("Generating embeddings...")
        requirement_embeddings = model.encode(requirements, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
        guideline_embeddings = model.encode(guidelines, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

        # Save embeddings to cache
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({
                "requirements": requirements,
                "requirement_embeddings": requirement_embeddings,
                "guidelines": guidelines,
                "guideline_embeddings": guideline_embeddings
            }, fOut)
        print("Embeddings saved to cache.")
    else:
        print("Loading embeddings from cache...")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            requirements = cache_data["requirements"]
            requirement_embeddings = cache_data["requirement_embeddings"]
            guidelines = cache_data["guidelines"]
            guideline_embeddings = cache_data["guideline_embeddings"]

    return requirement_embeddings, guideline_embeddings


def run_pipeline_1(file_path_excel, file_paths_csv, embedding_cache_path, result_path, top_k=10):
    """
    Run Pipeline 1 for semantic search.

    Args:
        file_path_excel (str): Path to the Excel file containing guidelines.
        file_paths_csv (list): List of CSV file paths containing requirements.
        embedding_cache_path (str): Path to cache embeddings.
        result_path (str): Path to save results.
        top_k (int): Number of top matches to retrieve.

    Returns:
        None
    """
    # Ensure directories exist
    ensure_directories_exist(["embedding", "result"])

    # Load model
    model_name = "quora-distilbert-multilingual"
    model = SentenceTransformer(model_name)

    # Load and process data
    guidelines, requirements = load_and_process_data(file_path_excel, file_paths_csv)

    # Generate or load embeddings
    requirement_embeddings, guideline_embeddings = generate_embeddings(
        model, requirements, guidelines, embedding_cache_path, batch_size=32
    )

    print(f"Length of guideline_embeddings   : {len(guideline_embeddings)}")
    print(f"Length of requirement_embeddings : {len(requirement_embeddings)}")

    # Perform global semantic search
    print("Performing semantic search...")
    start_time = time()
    search_results = semantic_search_global(guidelines, requirement_embeddings, requirements, model, top_k=top_k)
    elapsed_time = time() - start_time
    print(f"Semantic search completed in {elapsed_time:.2f} seconds.")

    # Save results to CSV
    results_list = []
    for result in search_results:
        query = result["query"]
        for match in result["matches"]:
            results_list.append({"query": query, "matched_text": match["text"], "score": match["score"]})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(result_path, index=False)
    print(f"Results saved to '{result_path}'.")
