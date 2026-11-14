
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

def preprocess_text(sentence):
    # Tokenize the input sentence
    tokens = word_tokenize(sentence)

    # Remove English stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stop_words
    ]

    # Apply Porter stemming
    stemmer = PorterStemmer()
    stemmed_words = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    print("Original Tokens:", tokens)
    print("Tokens Without Stopwords:", filtered_tokens)
    print("Stemmed Words:", stemmed_words)

sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
preprocess_text(sentence)
