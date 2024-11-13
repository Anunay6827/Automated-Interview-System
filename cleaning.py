import pandas as pd
import re

file_path = '/content/dataset.csv'  #replace with the actual path
data = pd.read_csv(file_path, encoding='latin1')

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

if 'question' in data.columns and 'answer' in data.columns:
    data['question'] = data['question'].apply(clean_text)
    data['answer'] = data['answer'].apply(clean_text)
else:
    print("The dataset does not contain 'question' and 'answer' columns.")

from IPython.display import display
display(data)

cleaned_file_path = '/content/cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}")
