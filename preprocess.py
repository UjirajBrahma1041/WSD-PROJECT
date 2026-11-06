# -*- coding: utf-8 -*-
"""
Preprocess Module
-----------------
This module handles all the natural language processing (NLP) preprocessing steps
for Assamese text. It includes text cleaning, punctuation removal (specifically handling 
Assamese punctuation like the danda '।'), tokenization, and stopword removal.

Designed for the B.Tech CSE 6th Semester WSD Mini Project.
"""

import re

# Custom list of common Assamese stopwords to enhance sentence embedding quality.
# Stopwords are high-frequency words that carry little semantic weight. 
# Removing them helps context matching by focusing on meaningful content words.
ASSAMESE_STOPWORDS = [
    "আৰু", "এই", "হয়", "পৰা", "লগত", "মই", "তুমি", "তেওঁ", "তাক", 
    "আছে", "হৈছে", "হৈ", "দি", "ৰখা", "থৈ", "বাবে", "লৈ", 
    "টো", "খন", "জোপাৰ", "জোপাত", "বোৰ", "বোৰক", "ৰ", "ত", "এ", "ও"
]

# Set of punctuation characters to be removed from Assamese sentences
PUNCTUATION_REG = re.compile(r'[।\-\,\.\?\!\:\;\"\'\(\)\[\]\{\}\‘\’\“\”\…]')

def clean_text(text):
    """
    Cleans the input text by performing lowercasing (for mixed or English tokens)
    and removing digits (both English and Assamese).
    
    Args:
        text (str): Input Assamese sentence.
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase standard characters (mostly for any mixed English/Latin words)
    text = text.lower()
    
    # 2. Remove English digits (0-9) and Assamese digits (০-৯)
    text = re.sub(r'[0-9০১২৩৪৫৬৭৮৯]', '', text)
    
    # 3. Strip extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_punctuation(text):
    """
    Removes punctuation marks including standard punctuation and the Assamese '।' (danda).
    
    Args:
        text (str): Input text.
    Returns:
        str: Text without punctuation.
    """
    return PUNCTUATION_REG.sub(' ', text)

def tokenize(text):
    """
    Splits the sentence into individual tokens (words) using white spaces.
    
    Args:
        text (str): Preprocessed text.
    Returns:
        list of str: Tokenized words.
    """
    # Split by spaces and filter out empty strings
    return [word for word in text.split(' ') if word.strip()]

def remove_stopwords(tokens):
    """
    Filters out predefined Assamese stopwords from the token list.
    
    Args:
        tokens (list of str): List of tokenized words.
    Returns:
        list of str: Tokens without stopwords.
    """
    return [token for token in tokens if token not in ASSAMESE_STOPWORDS]

def preprocess_pipeline(text, filter_stopwords=True):
    """
    A complete NLP pipeline that combines all preprocessing steps:
    1. Cleaning (lowercasing, digits removal)
    2. Punctuation removal
    3. Tokenization
    4. Stopword filtering (optional)
    
    Args:
        text (str): Raw input Assamese text.
        filter_stopwords (bool): Whether to remove stopwords. Default is True.
    Returns:
        list of str: Preprocessed, tokenized, and cleaned words.
    """
    cleaned = clean_text(text)
    no_punct = remove_punctuation(cleaned)
    tokens = tokenize(no_punct)
    
    if filter_stopwords:
        tokens = remove_stopwords(tokens)
        
    return tokens

# Small unit test when executed directly
if __name__ == "__main__":
    sample = "বতাহত গছৰ পাত সৰি পৰিছে।"
    print("Original Text:", sample)
    print("Preprocessed Tokens:", preprocess_pipeline(sample))
