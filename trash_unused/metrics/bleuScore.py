#bleu score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_score(reference, candidate):
    """
    Compute BLEU score for a candidate text against a reference.
    
    Args:
        reference (str): Reference text.
        candidate (str): Candidate text.
    
    Returns:
        float: BLEU score (0 to 1).
    """
    reference_tokens = [reference.lower().split()]
    candidate_tokens = candidate.lower().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
