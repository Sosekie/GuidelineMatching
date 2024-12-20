import asyncio
import openai
import time
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from pipeline.data_processing import load_and_process_data, ensure_directories_exist

# Set the API client
client = openai.AsyncOpenAI(
    api_key = "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"
)

with open('prompts/prompt_system.txt', 'r', encoding='utf-8') as file:
    system_content = file.read()


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
    query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)

    M, N = len(queries), len(corpus_embeddings)
    bi_matrix = np.zeros((M, N))

    for idx, query_hits in enumerate(hits):
        for hit in query_hits: 
            bi_matrix[idx][hit["corpus_id"]] = hit["score"]

    np.save("bi_matrix.npy", bi_matrix)

    return bi_matrix


async def fetch_response(sentence, guideline, model_name):
    """异步发送单个 API 请求，判断是否匹配"""
    question_content = f"<sentence>{sentence}</sentence><guideline>{guideline}</guideline>"
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": question_content},
            ]
        )
        result = int(response.choices[0].message.content[-1])  # 假设返回值是"0"或"1"
        # await asyncio.sleep(0.2)
        return result
    except Exception as e:
        print(f"Error processing sentence '{sentence}' with guideline '{guideline}': {e}")
        return 0  # 默认返回0表示不匹配


async def batched_requests(tasks, batch_size):
    results = []
    for i in range(0, len(tasks), batch_size):
        print(f"batch {i}")
        start_time = time.time()
        batch = tasks[i:i + batch_size]
        results.extend(await asyncio.gather(*batch))
        # 计算批次完成时间
        elapsed_time = time.time() - start_time
        print(f"Completed in {elapsed_time:.2f} seconds.")
        
        # 如果完成时间小于指定间隔，等待剩余时间
        if elapsed_time < 30:
            wait_time = 30 - elapsed_time
            print(f"Waiting for {wait_time:.2f} seconds before starting the next batch.")
            await asyncio.sleep(wait_time)

    return results


async def main(sentences, guidelines, bi_matrix, model_name):
    """
    根据 bi_matrix 的条件筛选参与 fetch_response 的任务，并更新矩阵
    """
    M, N = bi_matrix.shape
    matrix = np.zeros((M, N), dtype=int)  # 初始化结果矩阵，数据类型为整数

    tasks = []  # 存储任务
    indices = []  # 存储任务对应的 (i, j) 下标

    # 遍历 bi_matrix 筛选需要处理的 guideline
    for i, sentence in enumerate(sentences):
        for j, guideline in enumerate(guidelines):
            if bi_matrix[i, j] > 0:  # 筛选 bi_matrix 对应值大于 0 的任务
                tasks.append(fetch_response(sentence, guideline, model_name))
                indices.append((i, j))

    # 批量执行 fetch_response，限制每批任务数量为 batch_size
    batch_size = 200
    task_results = await batched_requests(tasks, batch_size)

    # 将结果填充到对应的矩阵位置
    for (i, j), value in zip(indices, task_results):
        matrix[i, j] = value

    return matrix


# 运行程序
def run_pipeline_3(file_path_excel, file_paths_csv, embedding_cache_path, result_path, gpt_model_name):

    # Ensure directories exist
    ensure_directories_exist(["embedding", "result"])

    # Load models
    bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # Load and process data
    guidelines, sentences = load_and_process_data(file_path_excel, file_paths_csv)

    # Generate or load embeddings
    if not os.path.exists(embedding_cache_path):
        print("Generating embeddings...")
        corpus_embeddings = bi_encoder.encode(guidelines, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({
                "sentences": sentences,
                "corpus_embeddings": corpus_embeddings,
                "guidelines": guidelines
            }, fOut)
        print("Embeddings saved to cache.")
    else:
        print("Loading embeddings from cache...")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            sentences = cache_data["sentences"]
            corpus_embeddings = cache_data["corpus_embeddings"]
            guidelines = cache_data["guidelines"]

    # Perform Bi-Encoder retrieval
    print("Performing Bi-Encoder retrieval...")
    start_time = time.time()
    bi_matrix = bi_encoder_retrieve(bi_encoder, sentences, corpus_embeddings, guidelines, top_k=20)
    print(f"Bi-Encoder retrieval completed in {time.time() - start_time:.2f} seconds.")

    print("Performing gpt re-ranking...")
    start_time = time.time()
    matching_matrix = asyncio.run(main(sentences, guidelines, bi_matrix, gpt_model_name))
    print(f"gpt re-ranking completed in {time.time() - start_time:.2f} seconds.")

    # 保存为 .npy 文件
    np.save(result_path, matching_matrix)
    print(f"Matrix saved to {result_path}")

    print("Final Matrix:")
    print(matching_matrix)