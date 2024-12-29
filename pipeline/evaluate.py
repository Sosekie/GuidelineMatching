import numpy as np


def recall_k(matched_matrix, target_matrix, k=10):

    # Ensure n is within the valid range
    if not (7 <= k <= 20):
        raise ValueError("n must be between 7 and 20")
    
    M, N = matched_matrix.shape
    recall_matrix = np.zeros((M, N), dtype="int")
    
    # Construct recall_matrix based on the top-n values per row
    for i in range(M):
        # Get the indices of the top-n values in the row
        top_n_indices = np.argsort(matched_matrix[i])[-k:]  # Indices of the top-n largest values
        recall_matrix[i, top_n_indices] = 1  # Set those positions to 1

    total_score = []
    for i in range(M): 
        row_match = np.sum(target_matrix[i])
        if row_match == 0:
            # print(f"row_match is 0 at row {i}.")
            continue

        row_score = np.sum(recall_matrix[i] & target_matrix[i]) / row_match
        total_score.append(row_score)
    
    # Calculate the average score
    recall_score = np.mean(total_score)
    
    return recall_score


def unrelated_acc(matched_matrix):
    # 每一行求和
    row_sums = np.sum(matched_matrix, axis=1)

    # 矢量化处理
    result = (np.sum(row_sums[:30] > 0) + np.sum(row_sums[30:] == 0)) / len(row_sums)

    return result

import numpy as np
from scipy.stats import f_oneway

def f_distribution_detection(matched_matrix, target_matrix, alpha=0.05):
    """
    使用一元方差分析(ANOVA)的F检验来判断当前requirement是否与guideline有显著关系。
    如果p_value >= alpha，不拒绝原假设 -> 判定“无GT”；
    如果p_value < alpha，拒绝原假设 -> 判定“有GT”。
    
    参数
    ----------
    matched_matrix: np.ndarray, shape=(M, N)
        模型预测的相似度矩阵，M 是 requirement 个数，N 是 guideline 个数
    target_matrix: np.ndarray, shape=(M, N)
        真实的标签矩阵，0/1 标记
    alpha: float
        显著性水平（常用0.05或0.01等）

    返回
    ----------
    accuracy: float
        针对“有无GT”预测的准确率
    """

    M, N = matched_matrix.shape
    if target_matrix.shape != (M, N):
        raise ValueError("target_matrix 形状与 matched_matrix 不一致!")
    
    correct_count = 0

    # 将所有行(除了自己那行)的数据整合为一个大数组，方便后面做对比
    # 注意，对于很大的矩阵，该做法会在性能和内存上有较大开销。
    all_data_flatten = matched_matrix.flatten()

    for i in range(M):
        # 取第 i 行 (该 requirement 对所有 guideline 的相似度)
        row_data = matched_matrix[i]
        
        # 将第 i 行从整合后的数据里排除
        # mask 为 True 的表示要保留的数据
        mask = np.ones(M*N, dtype=bool)
        # 第 i 行在 flatten 后所在的区间 [i*N, (i+1)*N)
        mask[i*N:(i+1)*N] = False
        other_data = all_data_flatten[mask]

        # 使用一元方差分析(ANOVA)进行F检验
        # 注意：f_oneway 需要传入至少两组数据，这里我们分两组：
        #   1) row_data
        #   2) other_data
        # 若认为 other_data 是过于庞杂且分布不一致，也可以考虑进一步拆分、或者改用其他统计方法
        f_stat, p_value = f_oneway(row_data, other_data)

        # 预测：p_value 小于 alpha，说明拒绝原假设 => 认为本 requirement 有 GT
        pred_has_gt = (p_value < alpha)

        # 真实值：target_matrix[i] 中是否全为 0
        real_has_gt = (np.sum(target_matrix[i]) > 0)

        if pred_has_gt == real_has_gt:
            correct_count += 1
    
    # 计算准确率
    accuracy = correct_count / M
    return accuracy
