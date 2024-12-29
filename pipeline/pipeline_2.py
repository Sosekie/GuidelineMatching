import os
import pickle
import pandas as pd
import numpy as np
from time import time
import statistics

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from data_processing import load_and_process_data, ensure_directories_exist


def bi_encoder_retrieve(sentences_embeddings, guidelines_embeddings, top_k=20):
    hits = util.semantic_search(sentences_embeddings, guidelines_embeddings, top_k=top_k)

    M, N = len(sentences_embeddings), len(guidelines_embeddings)
    bi_matrix = np.zeros((M, N))

    for idx, query_hits in enumerate(hits):
        for hit in query_hits: 
            bi_matrix[idx][hit["corpus_id"]] = hit["score"]

    np.save("bi_matrix.npy", bi_matrix)
    return bi_matrix


def cross_encoder_rerank(model, bi_matrix, sentences, guidelines, top_k=20):
    rerank_matrix = np.zeros_like(bi_matrix)

    for i, row in enumerate(bi_matrix):
        cross_idx = np.where(row > 0)[0]
        cross_inp = [[sentences[i], guidelines[idx]] for idx in cross_idx]
        cross_scores = model.predict(cross_inp)
        top_idx = np.argsort(cross_scores)[-top_k:]

        rerank_matrix[i][cross_idx[top_idx]] = cross_scores[top_idx]

    return rerank_matrix


def run_pipeline_2(
    file_path_excel, 
    file_paths_csv, 
    embedding_cache_path, 
    result_path, 
    top_k=20, 
    bi_top_k=40
):
    """
    Run Pipeline 2 for semantic search with reranking.

    Args:
        file_path_excel (str): Path to the Excel file containing guidelines.
        file_paths_csv (list): List of CSV file paths containing requirements.
        embedding_cache_path (str): Path to cache embeddings.
        result_path (str): Path to save results (.npy).
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
    guidelines, sentences = load_and_process_data(file_path_excel, file_paths_csv)

    # Generate or load embeddings
    if not os.path.exists(embedding_cache_path):
        print("Generating embeddings...")
        sentences_embeddings = bi_encoder.encode(
            sentences, batch_size=32, convert_to_tensor=True, show_progress_bar=True
        )
        guidelines_embeddings = bi_encoder.encode(
            guidelines, batch_size=32, convert_to_tensor=True, show_progress_bar=True
        )
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({
                "sentences": sentences,
                "sentences_embeddings": sentences_embeddings,
                "guidelines": guidelines, 
                "guidelines_embeddings": guidelines_embeddings
            }, fOut)
        print("Embeddings saved to cache.")
    else:
        print("Loading embeddings from cache...")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            sentences = cache_data["sentences"]
            sentences_embeddings = cache_data["sentences_embeddings"]
            guidelines = cache_data["guidelines"]
            guidelines_embeddings = cache_data["guidelines_embeddings"]

    # Perform Bi-Encoder retrieval
    print("Performing Bi-Encoder retrieval...")
    start_time = time()
    bi_matrix = bi_encoder_retrieve(sentences_embeddings, guidelines_embeddings, top_k=bi_top_k)
    bi_elapsed_time = time() - start_time
    print(f"Bi-Encoder retrieval completed in {bi_elapsed_time:.2f} seconds.")

    # Perform Cross-Encoder reranking
    print("Performing Cross-Encoder reranking...")
    start_time = time()
    rerank_matrix = cross_encoder_rerank(cross_encoder, bi_matrix, sentences, guidelines, top_k=top_k)
    cross_elapsed_time = time() - start_time
    print(f"Cross-Encoder reranking completed in {cross_elapsed_time:.2f} seconds.")

    # 保存为 .npy 文件
    np.save(result_path, rerank_matrix)
    print(f"Matrix saved to {result_path}")
    print(rerank_matrix.shape)


def main():
    # Define file paths
    file_path_excel = "2024-09-27_Food_Waren-und_Dienstleistungsgruppe_V_4.0.xlsx"
    file_paths_csv = "grouped_result.csv"
    embedding_cache_path = "embedding/Pipeline_2_embeddings_cache.pkl"

    # 结果文件，将在循环里自动命名
    # result_path = "result/p2_matching_matrix.npy"

    # Set hyperparameters
    top_k = 20     # Number of top reranked matches to retrieve
    bi_top_k = 40  # Number of top Bi-Encoder matches before reranking

    repetition = 3
    time_list = []

    for run_idx in range(1, repetition + 1):
        print(f"===== Pipeline 2: Run {run_idx}/{repetition} =====")

        # 如果希望三次都生成新的 embeddings，可在此删除缓存
        # if os.path.exists(embedding_cache_path):
        #     os.remove(embedding_cache_path)
        #     print("Removed old cache to re-generate embeddings each run.")

        # 保存结果时带 run_idx
        result_path = f"result/p2_matching_matrix_{run_idx}.npy"

        start_time = time()
        run_pipeline_2(
            file_path_excel=file_path_excel,
            file_paths_csv=file_paths_csv,
            embedding_cache_path=embedding_cache_path,
            result_path=result_path,
            top_k=top_k,
            bi_top_k=bi_top_k
        )
        end_time = time()

        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
        print(f"[Run {run_idx}] 总耗时: {elapsed_time:.3f} 秒\n")

    # 计算平均时间和标准差
    mean_time = statistics.mean(time_list)
    std_time = statistics.pstdev(time_list)  # 如想要样本标准差，用 stdev

    # 格式化为 "xx.xxx±xx.xxx"
    time_summary_str = f"{mean_time:.3f}±{std_time:.3f}"

    # 写入到CSV
    ensure_directories_exist(["result"])
    time_csv_path = "result/p2_time.csv"
    import csv
    with open(time_csv_path, mode="w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 写入三次耗时
        writer.writerow(["run1", "run2", "run3"])
        writer.writerow(time_list)
        # 写入平均时间 ± 标准差
        writer.writerow(["mean±std", time_summary_str])

    print(f"三次运行时间: {time_list}")
    print(f"平均时间±标准差: {time_summary_str}")
    print(f"已写入 {time_csv_path}")


if __name__ == "__main__":
    main()
