import asyncio
import openai
import time
import os
import pickle
import csv
import statistics
import numpy as np
from sentence_transformers import SentenceTransformer, util
from data_processing import load_and_process_data, ensure_directories_exist

# Set the API client
client = openai.AsyncOpenAI(
    api_key = "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"
)

with open('prompts/prompt_system_cot.txt', 'r', encoding='utf-8') as file:
    system_content = file.read()


def bi_encoder_retrieve(model, queries, corpus_embeddings, corpus_texts, top_k=10):
    """
    Perform retrieval using a Bi-Encoder.
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


def get_gpt_result(response_content):
    for char in reversed(response_content[-20:]):
        if char == '"':  # stop when meet "
            return 0
        if char.isdigit():  # return number
            return int(char)
    return 0  # no number => return 0


async def fetch_response(sentence, guideline, model_name):
    """
    异步发送单个 API 请求，判断是否匹配
    """
    question_content = f"<sentence>{sentence}</sentence><guideline>{guideline}</guideline>"
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": question_content},
            ]
        )
        result = get_gpt_result(response.choices[0].message.content)  # 假设返回值是 "0" 或 "1"
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
        
        # 如果完成时间 < 60秒，等待剩余时间
        if elapsed_time < 60:
            wait_time = 60 - elapsed_time
            print(f"Waiting for {wait_time:.2f} seconds before starting the next batch.")
            await asyncio.sleep(wait_time)
    return results


async def main_async(sentences, guidelines, bi_matrix, model_name):
    """
    根据 bi_matrix 的条件筛选参与 fetch_response 的任务，并更新矩阵
    """
    M, N = bi_matrix.shape
    matrix = np.zeros((M, N), dtype=int)

    tasks = []
    indices = []

    # 遍历 bi_matrix，找出 >0 的单元格
    for i, sentence in enumerate(sentences):
        for j, guideline in enumerate(guidelines):
            if bi_matrix[i, j] > 0:
                tasks.append(fetch_response(sentence, guideline, model_name))
                indices.append((i, j))

    # 按批次并发请求
    batch_size = 200
    task_results = await batched_requests(tasks, batch_size)

    # 填充结果
    for (i, j), value in zip(indices, task_results):
        matrix[i, j] = value

    return matrix


def run_pipeline_4(file_path_excel, file_paths_csv, embedding_cache_path, result_path, gpt_model_name):
    """
    执行 Pipeline 4，对应 p4_matching_matrix_{run_idx}.npy
    """
    # 确保目录存在
    ensure_directories_exist(["embedding", "result"])

    # 加载 Bi-Encoder
    bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # 读取并处理数据
    guidelines, sentences = load_and_process_data(file_path_excel, file_paths_csv)

    # 生成或加载 embeddings
    if not os.path.exists(embedding_cache_path):
        print("Generating embeddings...")
        corpus_embeddings = bi_encoder.encode(
            guidelines, batch_size=32, convert_to_tensor=True, show_progress_bar=True
        )
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

    # Bi-Encoder 检索
    print("Performing Bi-Encoder retrieval...")
    start_time = time.time()
    bi_matrix = bi_encoder_retrieve(bi_encoder, sentences, corpus_embeddings, guidelines, top_k=20)
    elapsed_bi = time.time() - start_time
    print(f"Bi-Encoder retrieval completed in {elapsed_bi:.2f} seconds.")

    # GPT re-ranking
    print("Performing gpt re-ranking...")
    start_time = time.time()
    matching_matrix = asyncio.run(main_async(sentences, guidelines, bi_matrix, gpt_model_name))
    elapsed_gpt = time.time() - start_time
    print(f"gpt re-ranking completed in {elapsed_gpt:.2f} seconds.")

    # 保存到 .npy
    np.save(result_path, matching_matrix)
    print(f"Matrix saved to {result_path}")
    print("Final Matrix:")
    print(matching_matrix)


def main():
    """
    做三次重复实验，统计耗时，并把结果矩阵保存成 p4_matching_matrix_{run_idx}.npy
    同时将耗时写入 result/p4_time.csv
    """
    # 定义路径
    file_path_excel = "2024-09-27_Food_Waren-und_Dienstleistungsgruppe_V_4.0.xlsx"
    file_paths_csv = "grouped_result.csv"
    embedding_cache_path = "embedding/Pipeline_4_embeddings_cache.pkl"

    # 指定 GPT 模型名称
    gpt_model_name = "gpt-4o-mini"  # 你自己的 GPT 模型

    # 三次重复实验
    repetition = 3
    time_list = []

    for run_idx in range(1, repetition + 1):
        print(f"===== Pipeline 4: Run {run_idx}/{repetition} =====")

        # 如果每次都想重新生成 embedding，可以解注释:
        # if os.path.exists(embedding_cache_path):
        #     os.remove(embedding_cache_path)
        #     print("Removed cache to re-generate embeddings each run.")

        # 结果文件名带序号
        result_path = f"result/p4_matching_matrix_{run_idx}.npy"

        start_time = time.time()
        run_pipeline_4(
            file_path_excel=file_path_excel,
            file_paths_csv=file_paths_csv,
            embedding_cache_path=embedding_cache_path,
            result_path=result_path,
            gpt_model_name=gpt_model_name
        )
        end_time = time.time()

        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
        print(f"[Run {run_idx}] 总耗时: {elapsed_time:.3f} 秒\n")

    # 计算平均时间和标准差
    mean_time = statistics.mean(time_list)
    std_time = statistics.pstdev(time_list)  # 若需样本标准差，可改用 stdev

    # 生成 "xx.xxx±xx.xxx" 格式
    time_summary_str = f"{mean_time:.3f}±{std_time:.3f}"

    # 写入CSV
    ensure_directories_exist(["result"])
    time_csv_path = "result/p4_time.csv"
    with open(time_csv_path, mode="w", newline='', encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Run1", "Run2", "Run3"])
        writer.writerow(time_list)
        writer.writerow(["mean±std", time_summary_str])

    print(f"三次运行时间: {time_list}")
    print(f"平均时间±标准差: {time_summary_str}")
    print(f"已写入 {time_csv_path}")


if __name__ == "__main__":
    main()
