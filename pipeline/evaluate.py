def recall_n(matched_sentences, target, n=5):
    """
    This function calculates the recall score for matched sentences against target sentences.
    
    Args:
        matched_sentences (list, shape [number of queries, 10]): A list where each element is a list of 10 matched sentences for a query.
        target (list, shape [number of queries, 1]): A list where each element is a list containing the target sentence for a query.
        n (int): The number of top sentences to consider for matching (between 1 and 10).
    
    Returns:
        float: The average recall score, calculated as the number of queries where the target sentence is found in the top n matched sentences divided by the total number of queries.
    """
    # Ensure n is within the valid range
    if not (1 <= n <= 10):
        raise ValueError("n must be between 1 and 10")
    
    total_queries = len(matched_sentences)
    total_score = 0
    
    # Iterate through each query's matched_sentences and target
    for i in range(total_queries):
        # Get the top n sentences for the current query
        top_n_sentences = matched_sentences[i][:n]
        # Get the corresponding target sentence
        target_sentence = target[i][0]
        
        # If target_sentence is in the top n sentences, score is 1
        if target_sentence in top_n_sentences:
            total_score += 1
    
    # Calculate the average score
    recall_score = total_score / total_queries
    
    return recall_score
