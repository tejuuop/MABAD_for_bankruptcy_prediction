import pandas as pd
import numpy as np
import os
import pickle
from preprocessing.finbert_preprocessor import FinBERTPreprocessor

def extract_and_save():
    print("Loading textual dataset...")
    # Load the raw text file
    data = pd.read_csv('data/mda_auditor.csv')
    
    # Initialize FinBERT on the GPU
    preprocessor = FinBERTPreprocessor(device='cuda:0')
    
    print("Extracting full documents with sliding window...")
    print("WARNING: This will take several hours. Let it run overnight if needed.")
    X_mda, X_auditor = preprocessor.transform(data)
    
    print("Mapping embeddings to UIDs to prevent shuffling errors...")
    # Map each uid directly to its 768-dimensional vector
    mda_dict = {uid: emb for uid, emb in zip(data['uid'], X_mda)}
    auditor_dict = {uid: emb for uid, emb in zip(data['uid'], X_auditor)}
    
    # Save the mapped dictionaries
    os.makedirs('data/embeddings', exist_ok=True)
    with open('data/embeddings/finbert_mda.pkl', 'wb') as f:
        pickle.dump(mda_dict, f)
    with open('data/embeddings/finbert_auditor.pkl', 'wb') as f:
        pickle.dump(auditor_dict, f)
        
    print("Done! Embeddings are securely locked to their UIDs and saved.")

if __name__ == '__main__':
    extract_and_save()