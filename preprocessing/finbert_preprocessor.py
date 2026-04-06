import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from preprocessing.base_preprocessor import BasePreprocessor
from utils.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

class FinBERTPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading FinBERT on {self.device}...")
        
        # Download/Load the pre-trained financial BERT model
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert").to(self.device)
        self.model.eval()

    def _get_embeddings(self, text_series, batch_size=16):
        """Extracts embeddings using overlapping chunks (Sliding Window) to preserve context."""
        texts = text_series.fillna("").tolist()
        final_document_embeddings = []
        
        logger.info(f"Extracting full-document embeddings with 50-word sliding window...")
        
        for text in tqdm(texts):
            words = text.split()
            chunk_size = 400
            step_size = 350 # Creates a 50-word overlap to preserve sentence boundaries
            chunks = []
            
            # Handle completely empty documents
            if not words:
                chunks = [""]
            else:
                # 1. Create the overlapping chunks
                for i in range(0, len(words), step_size):
                    chunk_text = ' '.join(words[i : i + chunk_size])
                    chunks.append(chunk_text)
                    # Stop if we have reached the end of the document
                    if i + chunk_size >= len(words):
                        break
            
            chunk_embeddings = []
            
            # 2. Process chunks in batches to save GPU memory
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                
                encoded_input = self.tokenizer(
                    batch_chunks, padding=True, truncation=True, max_length=512, return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    batch_embs = model_output.last_hidden_state[:, 0, :]
                    chunk_embeddings.append(batch_embs.cpu().numpy())
                    
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 3. Pool the overlapping chunks into a single 768-dim document representation
            document_average = np.vstack(chunk_embeddings).mean(axis=0)
            final_document_embeddings.append(document_average)
                
        return np.vstack(final_document_embeddings)
    
    def fit_transform(self, X):
        return self.transform(X)
       
    def transform(self, X):
        import os, pickle
        
        mda_path = 'data/embeddings/finbert_mda.pkl'
        aud_path = 'data/embeddings/finbert_auditor.pkl'
        
        # 1. If we already precomputed the embeddings, load them instantly!
        if os.path.exists(mda_path) and os.path.exists(aud_path):
            logger.info("Loading precomputed FinBERT embeddings from disk...")
            with open(mda_path, 'rb') as f:
                mda_dict = pickle.load(f)
            with open(aud_path, 'rb') as f:
                aud_dict = pickle.load(f)
                
            # Pull the correct embedding for the specific UIDs in this batch
            # X.index contains the UIDs because of how load_data.py sets the index
            X_mda = np.array([mda_dict[uid] for uid in X.index])
            X_auditor = np.array([aud_dict[uid] for uid in X.index])
            
            return X_mda, X_auditor
            
        # 2. If the files don't exist yet, extract them using the GPU
        logger.info("Extracting FinBERT embeddings for MD&A...")
        X_mda = self._get_embeddings(X['item_7'])
        
        logger.info("Extracting FinBERT embeddings for Auditor text...")
        X_auditor = self._get_embeddings(X['opinion_text'])
        
        return X_mda, X_auditor

    def save(self):
        logger.info("FinBERT preprocessor relies on pre-trained weights. Nothing to save locally.")

    def load(self):
        pass