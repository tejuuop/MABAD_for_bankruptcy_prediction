from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.standardizer_preprocessor import Standardizer
from preprocessing.pass_preproccessor import PASSpreprocessor

from sklearn.impute import SimpleImputer
import pickle
import os


class MeanImputer(BasePreprocessor):

    def __init__(self, **kwargs):

        strategy = kwargs['imputer']
        if strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        else:
            self.imputer = SimpleImputer(
                strategy='constant', fill_value=int(strategy))
        
        self._standardizer_preprocessor = Standardizer() if kwargs['standardize'] else PASSpreprocessor()


    def fit_transform(self, X):
        X = self.imputer.fit_transform(X)
        X = self._standardizer_preprocessor.fit_transform(X)
        return X

    def transform(self, X):
        X = self.imputer.transform(X)
        X = self._standardizer_preprocessor.transform(X)
        return X 

    def save(self):

        with open(os.path.join('preprocessing', 'saved_components', 'imputers', 'mean_imputer.pkl'), 'wb') as f:
            pickle.dump(self.imputer, f)
        
        self._standardizer_preprocessor.save()

    def load(self):

        with open(os.path.join('preprocessing', 'saved_components', 'imputers', 'mean_imputer.pkl'), 'rb') as f:
            self.imputer = pickle.load(f)
        
        self._standardizer_preprocessor.load()