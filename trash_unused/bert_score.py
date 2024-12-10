from bert_score import score

# Example summaries
generated_summary = "T5 is great for text summarization and NLP tasks."
reference_summary = "T5 performs well for summarization tasks."

# Compute BERTScore
P, R, F1 = score([generated_summary], [reference_summary], lang="en", verbose=True)
print("BERTScore (Precision, Recall, F1):", P.mean().item(), R.mean().item(), F1.mean().item())
