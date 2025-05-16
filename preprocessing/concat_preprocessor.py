from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.tfidf_preprocessor import TFIDFpreprocessor
import numpy as np

class CONCATpreprocessor(BasePreprocessor):

    def __init__(self,**kargs):
        self._tfidf_preprocessor = TFIDFpreprocessor(**kargs)

    def fit_transform(self, X):
        X_financial, X_mda_auditor = X
        X_mda_auditor = self._tfidf_preprocessor.fit_transform(X_mda_auditor)
        X = np.concatenate([X_financial, X_mda_auditor], axis=1)
        return X
    
    def transform(self, X):
        X_financial, X_mda_auditor = X
        X_mda_auditor = self._tfidf_preprocessor.transform(X_mda_auditor)
        X = np.concatenate([X_financial, X_mda_auditor], axis=1)
        return X

    
    def save(self):
        self._tfidf_preprocessor.save()

    
    def load(self):
        self._tfidf_preprocessor.load()