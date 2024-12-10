#relevance
from sklearn.metrics.pairwise import cosine_similarity

def calculate_relevance_score(query_embedding, candidate_embedding):
    """
    Compute relevance score as cosine similarity between query and candidate embeddings.
    
    Args:
        query_embedding (np.ndarray): Embedding vector for the query.
        candidate_embedding (np.ndarray): Embedding vector for the candidate question.
    
    Returns:
        float: Relevance score (0 to 1).
    """
    query_embedding = query_embedding.reshape(1, -1)
    candidate_embedding = candidate_embedding.reshape(1, -1)
    return cosine_similarity(query_embedding, candidate_embedding)[0][0]
