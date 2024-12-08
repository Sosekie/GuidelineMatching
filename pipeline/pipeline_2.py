import os
import pickle
import pandas as pd
from time import time
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pipeline.data_processing import load_and_process_data, ensure_directories_exist


def bi_encoder_retrieve(model, queries, corpus_embeddings, corpus_texts, top_k=10):
    """
    Perform retrieval using a Bi-Encoder.

    Args:
        model (SentenceTransformer): Bi-Encoder model for semantic search.
        queries (list): List of input queries.
        corpus_embeddings (tensor): Precomputed embeddings of the corpus.
        corpus_texts (list): Corresponding texts of the corpus embeddings.
        top_k (int): Number of top matches to retrieve.

    Returns:
        list: List of dictionaries containing top-k results for each query.
    """
    results = []
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)

    for idx, query_hits in enumerate(hits):
        query_result = {
            "query": queries[idx],
            "matches": [{"corpus_id": hit["corpus_id"], "score": hit["score"]} for hit in query_hits]
        }
        results.append(query_result)

    return results


def cross_encoder_rerank(model, bi_encoder_results, corpus_texts, top_k=10):
    """
    Re-rank Bi-Encoder results using a Cross-Encoder.

    Args:
        model (CrossEncoder): Cross-Encoder model for reranking.
        bi_encoder_results (list): Results from Bi-Encoder retrieval.
        corpus_texts (list): Corresponding texts of the corpus embeddings.
        top_k (int): Number of top reranked matches to retrieve.

    Returns:
        list: List of dictionaries containing reranked results for each query.
    """
    reranked_results = []

    for result in bi_encoder_results:
        query = result["query"]
        matches = result["matches"]

        # Prepare input for Cross-Encoder
        cross_inp = [[query, corpus_texts[match["corpus_id"]]] for match in matches]
        cross_scores = model.predict(cross_inp)

        # Add cross scores to matches
        for idx, match in enumerate(matches):
            match["cross_score"] = cross_scores[idx]

        # Sort by cross scores and limit to top_k
        reranked_results.append({
            "query": query,
            "matches": sorted(matches, key=lambda x: x["cross_score"], reverse=True)[:top_k]
        })

    return reranked_results


def run_pipeline_2(file_path_excel, file_paths_csv, embedding_cache_path, result_path, top_k=10, bi_top_k=32):
    """
    Run Pipeline 2 for semantic search with reranking.

    Args:
        file_path_excel (str): Path to the Excel file containing guidelines.
        file_paths_csv (list): List of CSV file paths containing requirements.
        embedding_cache_path (str): Path to cache embeddings.
        result_path (str): Path to save results.
        top_k (int): Number of top reranked matches to retrieve.
        bi_top_k (int): Number of top matches retrieved by Bi-Encoder before reranking.

    Returns:
        None
    """
    # Ensure directories exist
    ensure_directories_exist(["embedding", "result"])

    # Load models
    bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Load and process data
    guidelines, requirements = load_and_process_data(file_path_excel, file_paths_csv)

    # Generate or load embeddings
    if not os.path.exists(embedding_cache_path):
        print("Generating embeddings...")
        corpus_embeddings = bi_encoder.encode(requirements, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({
                "requirements": requirements,
                "corpus_embeddings": corpus_embeddings,
                "guidelines": guidelines
            }, fOut)
        print("Embeddings saved to cache.")
    else:
        print("Loading embeddings from cache...")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            requirements = cache_data["requirements"]
            corpus_embeddings = cache_data["corpus_embeddings"]
            guidelines = cache_data["guidelines"]

    # Perform Bi-Encoder retrieval
    print("Performing Bi-Encoder retrieval...")
    start_time = time()
    bi_encoder_results = bi_encoder_retrieve(bi_encoder, guidelines, corpus_embeddings, requirements, top_k=bi_top_k)
    print(f"Bi-Encoder retrieval completed in {time() - start_time:.2f} seconds.")

    # Perform Cross-Encoder reranking
    print("Performing Cross-Encoder reranking...")
    start_time = time()
    reranked_results = cross_encoder_rerank(cross_encoder, bi_encoder_results, requirements, top_k=top_k)
    print(f"Cross-Encoder reranking completed in {time() - start_time:.2f} seconds.")

    # Save results to CSV
    results_list = []
    for result in reranked_results:
        query = result["query"]
        for match in result["matches"]:
            results_list.append({
                "query": query,
                "matched_text": requirements[match["corpus_id"]],
                "bi_score": match["score"],
                "cross_score": match["cross_score"]
            })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(result_path, index=False)
    print(f"Results saved to '{result_path}'.")
