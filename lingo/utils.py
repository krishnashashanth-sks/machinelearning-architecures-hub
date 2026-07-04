from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): 
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub(r'[^a-z\s]', '', text) # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def preprocess_text_advanced(text):
    if not isinstance(text, str): # Ensure input is string
        return ""
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(processed_tokens)