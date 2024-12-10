from imports import *
from text_processing import *

model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute cosine similarity between user and question keyphrases
def compute_cosine_similarity(user_keyphrases, question_keyphrases):

    if not user_keyphrases or not question_keyphrases:
        return 0  

    user_str = ' '.join(user_keyphrases)
    question_str = ' '.join(question_keyphrases)

    if not user_str.strip() or not question_str.strip():
        return 0  

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([user_str, question_str])
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1]  

data['questions_keyphrases'] = data['questions_keyphrases'].apply(
    lambda x: x if isinstance(x, list) else x.split()
)

data = data[data['questions_keyphrases'].apply(len) > 0]

# Clean the text by converting to lowercase and removing non-alphanumeric characters
def clean_text(text):
    
    text = text.lower()  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  
    return text.strip()

# Check the completeness of an answer based on key phrases
def check_completeness(candidate_answer, key_phrases):
    key_text = " ".join(key_phrases)  
    candidate_answer_cleaned = clean_text(candidate_answer)  
    key_text_cleaned = clean_text(key_text)  
    tfidf = TfidfVectorizer().fit_transform([candidate_answer_cleaned, key_text_cleaned])  
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])  
    return similarity[0][0]

# Evaluate the accuracy of answer against the correct answer
def evaluate_accuracy(candidate_answer, correct_answer):

    if not isinstance(candidate_answer, str) or not isinstance(correct_answer, str):
        raise ValueError("Both candidate_answer and correct_answer must be strings.")

    embeddings = model.encode([candidate_answer, correct_answer], convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()  

# Evaluate the clarity of an answer based on the Flesch reading ease score
def evaluate_clarity(answer):
    score = textstat.flesch_reading_ease(answer)
    return score

# Evaluate the logical flow of an answer by calculating sentence count and average sentence length
def evaluate_logical_flow(answer):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(answer)
    sentence_count = len(list(doc.sents))
    avg_sentence_length = sum(len(sent) for sent in doc.sents) / sentence_count
    return {"sentence_count": sentence_count, "avg_sentence_length": avg_sentence_length}

# Compute cosine similarity between user and question keyphrases (repeated function)
def compute_cosine_similarity(user_keyphrases, question_keyphrases):
    user_str = ' '.join(user_keyphrases)
    question_str = ' '.join(question_keyphrases)
    vectorizer = CountVectorizer().fit_transform([user_str, question_str])
    similarity_matrix = cosine_similarity(vectorizer)
    return similarity_matrix[0][1]

data['questions_keyphrases'] = data['questions_keyphrases'].apply(
    lambda x: x if isinstance(x, list) else x.split()
)

user_intro = "supervised machine learning"
user_intro_keyphrases = user_intro.split()  

similarities = data['questions_keyphrases'].apply(
    lambda x: compute_cosine_similarity(user_intro_keyphrases, x)
)

most_similar_idx = similarities.idxmax()

# Evaluate the depth of an answer based on the coverage of extracted key phrases
def evaluate_depth_with_keyphrases(answer, extracted_keyphrases):
    if not extracted_keyphrases or not isinstance(extracted_keyphrases, list):
        return {"word_count": len(answer.split()), "keyphrase_coverage": 0, "missing_keyphrases": []}

    answer_lower = answer.lower()
    covered_keyphrases = [kw for kw in extracted_keyphrases if kw.lower() in answer_lower]
    keyphrase_coverage = len(covered_keyphrases) / len(extracted_keyphrases) if extracted_keyphrases else 0
    missing_keyphrases = [kw for kw in extracted_keyphrases if kw.lower() not in answer_lower]

    return {
        "word_count": len(answer.split()),
        "keyphrase_coverage": keyphrase_coverage,
        "covered_keyphrases": covered_keyphrases,
        "missing_keyphrases": missing_keyphrases
    }

# Calculate grammatical accuracy of a text using language tool
def calculate_grammatical_accuracy(text):

    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    num_errors = len(matches)
    total_words = len(text.split())

    if total_words == 0:
        return 1.0  

    error_rate = num_errors / total_words
    grammatical_accuracy = 1 - error_rate

    return grammatical_accuracy, num_error

# Evaluate the overall quality of an answer using multiple factors
def evaluate_answer(user_answer, correct_answer, keyphrases):
    
    if not user_answer.strip() or not correct_answer.strip():
        print("One of the answers is empty. Returning default scores.")
        return {
            'accuracy': 0.0,
            'completeness': 0.0,
            'clarity': 0.0,
            'logical_flow': 0.0
        }

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([user_answer, correct_answer])
    accuracy = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    keyphrases_count = sum(1 for kp in keyphrases if kp in user_answer.split())
    completeness = keyphrases_count / len(keyphrases) if keyphrases else 0

    clarity = 1.0  

    logical_flow = 1.0  

    return {
        'accuracy': accuracy,
        'completeness': completeness,
        'clarity': clarity,
        'logical_flow': logical_flow
    }

# Normalize the score to a percentage based on the total possible score
def normalize_score(total_weighted_grade, max_possible_score):
    try:
        return (total_weighted_grade / max_possible_score) * 100
    except ZeroDivisionError:
        return 0  

# Calculate the weighted grade based on various evaluation criteria
def calculate_weighted_grade(evaluation, answer_text):
    try:
        accuracy_score = float(evaluation.get('accuracy', 0))  
        completeness_score = float(evaluation.get('completeness', 0))
        clarity_score = float(evaluation.get('clarity', 0))
        grammatical_accuracy = float(evaluation.get('grammatical_accuracy', 0))

    except Exception as e:
        print(f"Error in conversion: {e}")
        return 0  

    print(f"Raw Scores - Accuracy: {accuracy_score}, Completeness: {completeness_score}, "
          f"Clarity: {clarity_score}, Grammatical Accuracy: {grammatical_accuracy}")

    weights = {
        'accuracy': 0.45,   
        'completeness': 0.25,
        'clarity': 0.1,
        'grammatical_accuracy': 0.2  
    }

    weighted_accuracy = accuracy_score * weights['accuracy']
    weighted_completeness = completeness_score * weights['completeness']
    weighted_clarity = clarity_score * weights['clarity']
    weighted_grammatical_accuracy = grammatical_accuracy * weights['grammatical_accuracy']

    total_weighted_grade = (weighted_accuracy + weighted_completeness + weighted_clarity + weighted_grammatical_accuracy)

    print(f"Weighted Total Score: {total_weighted_grade}")

    max_possible_score = sum(weights.values())  
    normalized_grade = normalize_score(total_weighted_grade, max_possible_score)

    print(f"Normalized Grade: {normalized_grade}")

    return normalized_grade