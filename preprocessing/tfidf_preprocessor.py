from preprocessing.base_preprocessor import BasePreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os

from utils.logger import  get_logger

logger = get_logger(__name__)

class TFIDFpreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):

        self.tfidf_dim = kwargs['tfidf_dim']
        self.ngram_range = kwargs['ngram_range']

        self.mda_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.tfidf_dim, stop_words = 'english', lowercase = True)
        self.auditor_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.tfidf_dim, stop_words = 'english', lowercase = True)


    def fit_transform(self, X):
        X_mda = self.mda_vectorizer.fit_transform(X['item_7'])
        X_auditor = self.auditor_vectorizer.fit_transform(X['opinion_text'])
        X = np.concatenate([X_mda.toarray(), X_auditor.toarray()], axis=1)
        return X
    
    def transform(self, X):
        X_mda = self.mda_vectorizer.transform(X['item_7'])
        X_auditor = self.auditor_vectorizer.transform(X['opinion_text'])
        X = np.concatenate([X_mda.toarray(), X_auditor.toarray()], axis=1)
        return X
    
    def save(self):
        mda_path = f"{os.path.join('preprocessing','saved_components','tfidf','mda_vectorizer.pkl')}"
        auditor_path = f"{os.path.join('preprocessing','saved_components','tfidf','auditor_vectorizer.pkl')}"

        with open(mda_path, 'wb') as file:
            pickle.dump(self.mda_vectorizer, file)

        with open(auditor_path, 'wb') as file:
            pickle.dump(self.auditor_vectorizer, file)

        logger.info("TFIDF vectorizers saved!")
            

    def load(self):
        mda_path = f"{os.path.join('preprocessing','saved_components','tfidf','mda_vectorizer.pkl')}"
        auditor_path = f"{os.path.join('preprocessing','saved_components','tfidf','auditor_vectorizer.pkl')}"

        with open(mda_path, 'rb') as file:
            self.mda_vectorizer = pickle.load(file)

        with open(auditor_path, 'rb') as file:
            self.auditor_vectorizer = pickle.load(file)       
        