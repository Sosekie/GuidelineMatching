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
            print(f"row_match is 0 at row {i}.")
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