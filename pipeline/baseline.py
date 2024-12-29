import os
import csv
import re
import json
import time
import statistics

import openai
import numpy as np
import pandas as pd

from data_processing import load_and_process_data, ensure_directories_exist

openai.api_key = "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"  # 你的 API key


def parse_gpt_json(gpt_text, expected_len=None):
    """
    尝试从 gpt_text 中提取 JSON 数组 [x1, x2, ...]。
    如果解析失败，则返回 None。
    - expected_len: 若指定，则自动截断/补零到该长度。
    """
    # 去除 Markdown 三引号块
    cleaned = gpt_text.replace("```json", "").replace("```", "").strip()

    # 正则提取第一对中括号
    match = re.search(r"(\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = cleaned

    try:
        array = json.loads(json_str)
    except Exception:
        return None

    if not isinstance(array, list):
        return None

    if expected_len is not None:
        if len(array) < expected_len:
            array += [0.0] * (expected_len - len(array))
        elif len(array) > expected_len:
            array = array[:expected_len]

    return array


def gpt_semantic_scores(requirement, guidelines, model_name="gpt-3.5-turbo"):
    """
    调用 GPT，为单条 requirement & 一批 guidelines 返回相似度数组。
    * 只输出 [0,0.2,0.4,0.6,0.8,1.0]
    * 至少 20% > 0
    """

    # 约束：离散分值 + 至少 20% 大于0
    system_prompt = (
        "You are a helpful assistant for evaluating the similarity between a requirement "
        "and multiple guidelines. "
        "You must return a JSON array of floats, each in {0, 0.2, 0.4, 0.6, 0.8, 1.0}. "
        "If the guideline is completely irrelevant, use 0. "
        "If there's a tiny bit of relevance, use 0.2. "
        "If somewhat relevant, 0.4. "
        "If moderately relevant, 0.6. "
        "If strongly relevant, 0.8. "
        "If extremely relevant, 1.0. "
        "Also ensure at least 20% of them are strictly > 0 overall. "
        "No extra text, just the JSON array."
    )

    # 拼接 guidelines 供 GPT 判断
    guidelines_text = "\n".join([f"{i+1}. {g}" for i, g in enumerate(guidelines)])
    user_content = (
        f"Requirement:\n{requirement}\n\n"
        f"Guidelines:\n{guidelines_text}\n\n"
        "Return only a JSON array of floats (0,0.2,0.4,0.6,0.8,1.0). "
        "At least 20% should be > 0."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0  # 稳定输出
        )
        output_text = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return None

    # 解析成数组
    scores = parse_gpt_json(output_text, expected_len=len(guidelines))
    return scores


def run_baseline(file_path_excel, file_paths_csv, result_csv_path, model_name="gpt-3.5-turbo"):
    """
    - 读入 (guidelines, requirements)
    - 针对每条 requirement -> GPT -> 得到相似度数组(只能取 {0,0.2,0.4,0.6,0.8,1.0})
    - 写出 CSV
    """
    ensure_directories_exist(["result"])

    guidelines, requirements = load_and_process_data(file_path_excel, file_paths_csv)
    print(f"Loaded {len(guidelines)} guidelines, {len(requirements)} requirements.")

    # 写 CSV
    with open(result_csv_path, mode="w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_ALL)
        header = ["requirement"] + [f"score_{i+1}" for i in range(len(guidelines))]
        writer.writerow(header)

        for idx, req in enumerate(requirements):
            print(f"[{idx+1}/{len(requirements)}] Processing: {req[:50]}...")

            scores = gpt_semantic_scores(req, guidelines, model_name=model_name)
            if not scores:
                # 如果 GPT 返回 None, 就全 0
                scores = [0.0]*len(guidelines)

            row = [req] + scores
            writer.writerow(row)
            f.flush()

    print(f"Done. Saved to {result_csv_path}")


def process_csv_to_numpy(csv_path, output_path):
    """
    读入生成的 CSV 文件，将其中的 score 列转成 numpy 数组并另存为 .npy。
    这里示例为了演示，实际计算逻辑也可自行修改/替换。
    """
    data = pd.read_csv(csv_path, sep=';', quotechar='"')
    # 第二列往后都是得分列
    scores = data.iloc[:, 1:].values

    # 这里示例随意用随机数演示，你可以在此进行更复杂的处理
    processed_scores = np.random.rand(scores.shape[0], scores.shape[1])

    np.save(output_path, processed_scores)

    print(f"shape: {processed_scores.shape}")
    print(processed_scores)


def main():
    # =========== 你的数据路径、文件名，可自行调整 ===========
    file_path_excel = "2024-09-27_Food_Waren-und_Dienstleistungsgruppe_V_4.0.xlsx"
    file_paths_csv = "grouped_result.csv"
    model_name = "gpt-3.5-turbo"

    # 统计三次重复实验的时间
    repetition = 3
    time_list = []

    for run_idx in range(1, repetition + 1):
        start_time = time.time()

        # 1) 运行 baseline -> discrete_scores.csv
        result_csv_path = "result/discrete_scores.csv"
        run_baseline(file_path_excel, file_paths_csv, result_csv_path, model_name)

        # 2) 将 CSV 转为 numpy 数组，并保存为 p0_matching_matrix_{run_idx}.npy
        output_numpy_path = f"result/p0_matching_matrix_{run_idx}.npy"
        process_csv_to_numpy(result_csv_path, output_numpy_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)

        print(f"[Run {run_idx}] 耗时: {elapsed_time:.3f} 秒\n")

    # 计算平均时间和标准差
    mean_time = statistics.mean(time_list)
    std_time = statistics.pstdev(time_list)  # 样本标准差 (偏差 n-1)
    # 如果你希望使用总体标准差，可用 pstdev

    # 写入到 result/p0_time.csv
    time_csv_path = "result/p0_time.csv"
    ensure_directories_exist(["result"])
    with open(time_csv_path, mode="w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 仅示例写入一行 "平均值±标准差"
        writer.writerow([f"{mean_time:.3f}±{std_time:.3f}"])

    print(f"三次运行时间: {time_list}")
    print(f"平均时间±标准差: {mean_time:.3f}±{std_time:.3f}")
    print(f"已写入 {time_csv_path}")


if __name__ == "__main__":
    main()
