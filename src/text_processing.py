from imports import *

# Clean text by removing unwanted characters, links, and formatting
def clean_text(text):
    if pd.isnull(text):
        return ""

    text = str(text)
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'http\S+\.(jpg|jpeg|png|gif|bmp|svg)', '', text, flags=re.IGNORECASE)  
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  
    text = re.sub(r'__(.*?)__', r'\1', text)  
    allowed_chars_pattern = r'[^a-zA-Z0-9.,?!:;+=\-\*/()\[\]{} ]+'  
    text = re.sub(allowed_chars_pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

if 'questions' in data.columns and 'answers' in data.columns:
    data['questions'] = data['questions'].apply(clean_text)
    data['answers'] = data['answers'].apply(clean_text)
else:
    print("The dataset does not contain 'question' and 'answer' columns.")

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)  
pd.set_option('display.max_colwidth', None)  

stop_words = set(stopwords.words('english'))

# Tokenize text by removing stopwords and punctuation
def clean_and_tokenize(text):

    if not isinstance(text, str):
        return []

    tokens = word_tokenize(text)

    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]

    return tokens

# Generate bigrams from tokenized text, or return unigrams if fewer than two tokens
def generate_bigrams_or_unigrams(tokens):
    if len(tokens) < 2:

        return [token for token in tokens]  

    bigram_list = list(bigrams(tokens))

    bigram_list = [[bigram[0], bigram[1]] for bigram in bigram_list if bigram[0] != bigram[1]]

    return bigram_list

data['question_tokens'] = data['questions'].apply(lambda x: generate_bigrams_or_unigrams(clean_and_tokenize(x)))