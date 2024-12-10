from nltk.translate.meteor_score import meteor_score

# Example summaries
reference = ["T5 performs well for summarization tasks."]
candidate = "T5 is great for text summarization and NLP tasks."

# Calculate METEOR score
meteor = meteor_score([reference], candidate)
print("METEOR Score:", meteor)
