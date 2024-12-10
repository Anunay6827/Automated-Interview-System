# Importing all necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import textstat
import spacy
import language_tool_python
import random

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('dataset.csv', encoding='latin1')