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

    for i in range(M): 
        total_match = np.sum(target_matrix[i])
        total_score = np.sum(recall_matrix[i] & target_matrix[i]) / total_match
    
    # Calculate the average score
    recall_score = total_score / N
    
    return recall_score
