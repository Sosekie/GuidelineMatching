import os
import pickle
import pandas as pd
import numpy as np
from time import time
import statistics
import csv

from sentence_transformers import SentenceTransformer, util
from data_processing import load_and_process_data, ensure_directories_exist


def semantic_search_global(sentences_embeddings, guidelines_embeddings, top_k=20):
    hits = util.semantic_search(sentences_embeddings, guidelines_embeddings, top_k=top_k)

    M, N = len(sentences_embeddings), len(guidelines_embeddings)
    match_matrix = np.zeros((M, N))

    for idx, query_hits in enumerate(hits):
        for hit in query_hits: 
            match_matrix[idx][hit["corpus_id"]] = hit["score"]

    return match_matrix


def run_pipeline_1(file_path_excel, file_paths_csv, embedding_cache_path, result_path, top_k=20):
    """
    Run Pipeline 1 for semantic search.

    Args:
        file_path_excel (str): Path to the Excel file containing guidelines.
        file_paths_csv (list): List of CSV file paths containing requirements.
        embedding_cache_path (str): Path to cache embeddings.
        result_path (str): Path to save results (.npy).

    Returns:
        None
    """
    # Ensure directories exist
    ensure_directories_exist(["embedding", "result"])

    # Load model
    model_name = "multi-qa-MiniLM-L6-cos-v1"
    model = SentenceTransformer(model_name)

    # Load and process data
    guidelines, sentences = load_and_process_data(file_path_excel, file_paths_csv)

    # Generate or load embeddings
    if not os.path.exists(embedding_cache_path):
        print("Generating embeddings...")
        sentences_embeddings = model.encode(sentences, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
        guidelines_embeddings = model.encode(guidelines, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
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

    # Perform global semantic search
    print("Performing semantic search...")
    start_time = time()
    match_matrix = semantic_search_global(sentences_embeddings, guidelines_embeddings, top_k=top_k)
    elapsed_time = time() - start_time
    print(f"Semantic search completed in {elapsed_time:.2f} seconds.")

    # 保存为 .npy 文件
    np.save(result_path, match_matrix)
    print(f"Matrix saved to {result_path}")

    print(match_matrix.shape)


def main():
    # 定义文件路径
    file_path_excel = "2024-09-27_Food_Waren-und_Dienstleistungsgruppe_V_4.0.xlsx"
    file_paths_csv = "grouped_result.csv"

    # 嵌入缓存路径
    embedding_cache_path = "embedding/Pipeline_1_embeddings_cache.pkl"

    # 重复实验的次数
    repetition = 3
    time_list = []

    for run_idx in range(1, repetition + 1):
        print(f"===== Pipeline 1: Run {run_idx}/{repetition} =====")

        # 结果文件
        result_path = f"result/p1_matching_matrix_{run_idx}.npy"

        # 如果你想每次都强制重新生成嵌入，可先删除缓存
        # if os.path.exists(embedding_cache_path):
        #     os.remove(embedding_cache_path)
        #     print("Removed old cache to ensure identical re-generation each run.")

        start_time = time()
        run_pipeline_1(
            file_path_excel=file_path_excel,
            file_paths_csv=file_paths_csv,
            embedding_cache_path=embedding_cache_path,
            result_path=result_path
        )
        end_time = time()
        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
        print(f"[Run {run_idx}] 耗时: {elapsed_time:.3f} 秒\n")

    # 计算平均时间和标准差
    mean_time = statistics.mean(time_list)
    # 如需样本标准差，可用 statistics.stdev(time_list)
    std_time = statistics.pstdev(time_list)

    # 生成 "xx.xxx±xx.xxx" 形式的字符串
    time_summary_str = f"{mean_time:.3f}±{std_time:.3f}"

    # 把时间信息写入 CSV
    ensure_directories_exist(["result"])
    time_csv_path = "result/p1_time.csv"
    with open(time_csv_path, mode="w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 写入三次运行的耗时
        writer.writerow(["run1", "run2", "run3"])
        writer.writerow(time_list)
        # 写入平均时间 ± 标准差
        writer.writerow(["mean±std", time_summary_str])

    print(f"三次运行时间: {time_list}")
    print(f"平均时间±标准差: {time_summary_str}")
    print(f"已写入 {time_csv_path}")


if __name__ == "__main__":
    main()
