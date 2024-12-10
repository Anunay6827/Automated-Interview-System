#semantic similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_semantic_similarity(query, candidate):
    """
    Compute semantic similarity between query and candidate texts using Sentence Transformers.
    
    Args:
        query (str): Query text.
        candidate (str): Candidate text.
    
    Returns:
        float: Semantic similarity score (0 to 1).
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Example of a small and fast model
    embeddings = model.encode([query, candidate])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
