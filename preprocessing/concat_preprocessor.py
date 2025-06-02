from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.tfidf_preprocessor import TFIDFpreprocessor
from preprocessing.mean_imputer_preprocessor import MeanImputer
from preprocessing.pass_preproccessor import PASSpreprocessor
from preprocessing.standardizer_preprocessor import Standardizer

import numpy as np

class CONCATpreprocessor(BasePreprocessor):

    def __init__(self,**kwargs):

        self._tfidf_preprocessor = TFIDFpreprocessor(**kwargs)
        self._imputer_preprocessor = MeanImputer(kwargs['imputer']) if kwargs['imputer'] else PASSpreprocessor()
        self._standardizer_preprocessor = Standardizer() if kwargs['standardize'] else PASSpreprocessor()

    def fit_transform(self, X):
        X_financial, X_mda_auditor = X
        X_mda_auditor = self._tfidf_preprocessor.fit_transform(X_mda_auditor)
        X_financial = self._imputer_preprocessor.fit_transform(X_financial)
        X = np.concatenate([X_financial, X_mda_auditor], axis=1)
        X = self._standardizer_preprocessor.fit_transform(X)
        return X
    
    def transform(self, X):
        X_financial, X_mda_auditor = X
        X_mda_auditor = self._tfidf_preprocessor.transform(X_mda_auditor)
        X_financial = self._imputer_preprocessor.transform(X_financial)
        X = np.concatenate([X_financial, X_mda_auditor], axis=1)
        X = self._standardizer_preprocessor.transform(X)
        return X

    
    def save(self):
        self._tfidf_preprocessor.save()
        self._imputer_preprocessor.save()
        self._standardizer_preprocessor.save()

    
    def load(self):
        self._tfidf_preprocessor.load()
        self._imputer_preprocessor.load()
        self._standardizer_preprocessor.load()