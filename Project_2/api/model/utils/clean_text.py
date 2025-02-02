# !python -m spacy download en_core_web_sm
import re
import nltk
import spacy

# Load spaCy's English model

nlp = spacy.load("en_core_web_sm")

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

def remove_stopwords(text):
    before = text
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_text = [word for word in words if word.lower() not in stop_words]
    text = " ".join(filtered_text)
    # print_change(before, text, "Remove Stopwords")
    return text

def lemmatize_text(text):
    before = text
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    # print_change(before, lemmatized_text, "Lemmatization")
    return lemmatized_text

def lowercase_text(text):
    before = text
    text = text.lower()
    # print_change(before, text, "Lowercasing")
    return text


def remove_whitespaces(text):
    before = text
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()  # Remove leading/trailing spaces
    text = text + '.'  # Append full stop
    # print_change(before, text, "Remove Whitespaces and Add Full Stop")
    return text

# 7. Removing URLs
def remove_urls(text):
    before = text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # print_change(before, text, "Remove URLs")
    return text

# 8. Replace Ampersand (&) with 'and' and Similar Substitutions
def replace_ampersand(text):
    before = text
    substitutions = {
        "&": "and",
        "%": "percent",
        "$": "dollar",
        "₹": 'rs.',
        "@": "",
        "*": "x",
        "#":'',
        '"': ' ',       
        "'s": ' ',      
        "'": '',       
        "_": ' ',       
        "=": ' ',       
        "|": ' ',
    }
    
    for old, new in substitutions.items():
        text = text.replace(old, new)
    text = re.sub(r'[©®™~^<>\\/`\[\]\(\)\{\}]', ' ', text)
    # print_change(before, text, "Replace Ampersand (&) and Similar Substitutions")
    return text

# 9. Replace Model Numbers or Part Numbers
def replace_model_numbers(text):
    before = text
    # Regex to match common model/part number patterns
    # Match sequences like 'ABC123', '123-XYZ', 'ABC-1234', etc.
    model_number_pattern = r'(?<!\s)([A-Za-z0-9]+(?:[-_\][A-Za-z0-9]+)+)(?!\s)'
    
    # Only replace model numbers with <MODEL>
    text = re.sub(model_number_pattern, lambda match: '<MODEL>' if any(c.isdigit() for c in match.group(0)) else match.group(0), text)
    
    # print_change(before, text, "Model")
    return text

def remove_repeated_phrases(text):
    # Split text into words
    words = text.split()

    # Keep track of seen phrases
    seen_phrases = set()

    # List to store words that are not repeated
    result = []

    # Iterate through words and construct phrases
    for i, word in enumerate(words):
        # Construct potential phrase by joining words
        phrase = ' '.join(words[i:i+1])  # Adjust the range for longer phrases if needed

        # Check if phrase is seen
        if phrase not in seen_phrases:
            result.append(word)
            seen_phrases.add(phrase)

    # Join the result list into a string
    return ' '.join(result)
# Combining All Preprocessing Steps
def preprocess_text(text):
    # print(f"Original Text: {text[:100]}...")  # Show original text (first 100 characters)
    
    # Call each function and apply transformations
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = lowercase_text(text)
    text = replace_ampersand(text)
    text = remove_whitespaces(text)
    text = remove_urls(text)
    text = replace_model_numbers(text)
    text = remove_repeated_phrases(text)
    
    # print(f"Processed Text: {text[:100]}...")  # Show processed text (first 100 characters)
    return text
