#rouge score
from rouge_score import rouge_scorer

def calculate_rouge_score(reference, candidate):
    """
    Compute ROUGE-L score for a candidate text against a reference.
    
    Args:
        reference (str): Reference text.
        candidate (str): Candidate text.
    
    Returns:
        float: ROUGE-L score (0 to 1).
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure
