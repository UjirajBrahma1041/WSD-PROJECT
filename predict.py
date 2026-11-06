# -*- coding: utf-8 -*-
"""
Predict Module
--------------
This script provides an interactive CLI-based prediction system for Assamese WSD.
It loads the trained Word2Vec model and preprocessed metadata, accepts user sentences,
and calculates cosine similarities to rank and predict the correct sense of 
ambiguous words like "পাত", "কৰ", and "মাটি".

Designed for the B.Tech CSE 6th Semester WSD Mini Project.
"""

import os
import pickle
import numpy as np
from preprocess import preprocess_pipeline
from gensim.models import Word2Vec

# Fallback for tabulate in case it's not installed yet
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Configurations
MODELS_DIR = 'models'
W2V_MODEL_PATH = os.path.join(MODELS_DIR, 'word2vec.model')
METADATA_PATH = os.path.join(MODELS_DIR, 'wsd_metadata.pkl')

def load_models():
    """
    Loads the trained Word2Vec model and precomputed metadata from disk.
    """
    if not os.path.exists(W2V_MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "Model files not found! Please run 'python train.py' first to train and save the models."
        )
    
    # Load Word2Vec model
    w2v_model = Word2Vec.load(W2V_MODEL_PATH)
    
    # Load training metadata
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
        
    return w2v_model, metadata['dataframe'], metadata['embeddings']

def get_sentence_embedding(tokens, model):
    """
    Generates sentence embedding by averaging word vectors.
    """
    vocab = model.wv
    valid_vectors = [vocab[word] for word in tokens if word in vocab]
    
    if not valid_vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(valid_vectors, axis=0)

def cosine_similarity(v1, v2):
    """
    Calculates cosine similarity between two vectors.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def predict_sense(input_sentence, ambiguous_word, model, df, embeddings, top_n=3):
    """
    Predicts the correct sense of the ambiguous word in the input sentence
    by ranking context sentences from the training dataset based on cosine similarity.
    
    Args:
        input_sentence (str): User input in Assamese.
        ambiguous_word (str): Target ambiguous word.
        model (Word2Vec): Word2Vec model.
        df (pd.DataFrame): Training dataset dataframe.
        embeddings (list of np.ndarray): Precalculated dataset embeddings.
        top_n (int): Number of top matches to return.
    Returns:
        dict: Prediction results containing predicted sense, category, and ranking table.
    """
    # 1. Preprocess the input sentence
    tokens = preprocess_pipeline(input_sentence, filter_stopwords=True)
    
    # Check if we have OOV (Out Of Vocabulary) situation
    vocab = model.wv
    oov_tokens = [tok for tok in tokens if tok not in vocab]
    in_vocab_tokens = [tok for tok in tokens if tok in vocab]
    
    # 2. Generate input sentence embedding
    input_vector = get_sentence_embedding(tokens, model)
    
    if np.all(input_vector == 0):
        return {
            'success': False,
            'reason': "Out of Vocabulary: None of the content words in your sentence exist in the model's vocabulary. Please use words present in the training set."
        }
    
    # 3. Filter training instances for the selected ambiguous word
    target_indices = df[df['ambiguous_word'] == ambiguous_word].index
    
    rankings = []
    
    # 4. Compute cosine similarity with each context sentence of the same target word
    for idx in target_indices:
        dataset_row = df.loc[idx]
        dataset_vector = embeddings[idx]
        
        sim = cosine_similarity(input_vector, dataset_vector)
        
        rankings.append({
            'sentence': dataset_row['sentence'],
            'sense': dataset_row['sense'],
            'category': dataset_row['category'],
            'similarity': sim
        })
        
    # Sort rankings in descending order of similarity
    rankings = sorted(rankings, key=lambda x: x['similarity'], reverse=True)
    
    # 5. Extract best match
    best_match = rankings[0]
    
    return {
        'success': True,
        'predicted_sense': best_match['sense'],
        'predicted_category': best_match['category'],
        'confidence': best_match['similarity'],
        'rankings': rankings[:top_n],
        'tokens_used': in_vocab_tokens,
        'oov_tokens': oov_tokens
    }

def print_table(headers, rows):
    """
    Helper to print a beautiful, aligned text-based table.
    Uses tabulate if available, else falls back to custom manual formatting.
    """
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    else:
        # Custom manual table layout
        print("-" * 80)
        header_str = " | ".join(f"{h:<18}" for h in headers)
        print(header_str)
        print("-" * 80)
        for row in rows:
            row_str = " | ".join(f"{str(item)[:18]:<18}" for item in row)
            print(row_str)
        print("-" * 80)

def main():
    print("==================================================================")
    print("  Assamese Word Sense Disambiguation (WSD) - Interactive Predictor")
    print("==================================================================")
    
    try:
        w2v_model, df, embeddings = load_models()
        print("[System] Models loaded successfully from disk.")
    except Exception as e:
        print(f"\n[Error] {e}")
        print("Please train the model first by running: python train.py\n")
        return
        
    ambiguous_words = list(df['ambiguous_word'].unique())
    
    while True:
        print("\n" + "="*50)
        print("Select an ambiguous Assamese word to disambiguate:")
        for idx, word in enumerate(ambiguous_words, start=1):
            print(f"{idx}. {word}")
        print(f"{len(ambiguous_words) + 1}. Exit System")
        print("="*50)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == str(len(ambiguous_words) + 1):
            print("\nThank you for using the Assamese WSD System! Have a great day.")
            break
            
        if choice not in [str(i) for i in range(1, len(ambiguous_words) + 1)]:
            print("[Warning] Invalid choice! Please enter a valid number.")
            continue
            
        selected_word = ambiguous_words[int(choice) - 1]
        print(f"\nYou selected the ambiguous word: '{selected_word}'")
        
        print("\nEnter your Assamese sentence below.")
        print(f"Tip: Ensure the sentence provides a clear context around the word '{selected_word}'.")
        print("Example: 'বতাহত গছৰ পাত সৰি পৰিছে।'")
        
        input_sentence = input("Assamese Input Sentence: ").strip()
        
        if not input_sentence:
            print("[Warning] Sentence cannot be empty. Please try again.")
            continue
            
        # Optional Warning if the word is not explicitly in the sentence
        if selected_word not in input_sentence:
            print(f"\n[Note] Your sentence does not explicitly contain the word '{selected_word}'.")
            print("Running prediction based on semantic contextual similarity...")
            
        # Run prediction
        print("\nProcessing sentence and calculating semantic vectors...")
        result = predict_sense(input_sentence, selected_word, w2v_model, df, embeddings)
        
        if not result['success']:
            print(f"\n[Error] {result['reason']}")
            continue
            
        print("\n" + "*"*60)
        print("                     DISAMBIGUATION RESULT")
        print("*"*60)
        print(f"Input Sentence :  {input_sentence}")
        print(f"Target Word    :  {selected_word}")
        print(f"Predicted Sense:  {result['predicted_sense']}")
        print(f"Category       :  {result['predicted_category']}")
        print(f"Confidence     :  {result['confidence'] * 100:.2f}% (Cosine Similarity)")
        print(f"Tokens Matched :  {result['tokens_used']}")
        if result['oov_tokens']:
            print(f"Tokens Ignored :  {result['oov_tokens']} (Out of Vocabulary)")
        print("*"*60)
        
        # Display Rankings
        print("\nTop 3 Context Similarity Rankings:")
        headers = ["Rank", "Context Sentence", "Matched Sense", "Category", "Similarity (%)"]
        rows = []
        for i, match in enumerate(result['rankings'], start=1):
            rows.append([
                f"#{i}",
                match['sentence'],
                match['sense'],
                match['category'],
                f"{match['similarity'] * 100:.2f}%"
            ])
            
        print_table(headers, rows)
        
        # Ask to continue
        input("\nPress ENTER to continue...")

if __name__ == "__main__":
    main()
