from preprocessing.base_preprocessor import BasePreprocessor
from sklearn.impute import SimpleImputer
import pickle
import os


class ImputerPreprocessor(BasePreprocessor):

    def __init__(self, **kwargs):

        strategy = kwargs['imputer']

        if strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        else:
            self.imputer = SimpleImputer(
                strategy='constant', fill_value=int(strategy))
            
    def fit_transform(self, X):
        X = self.imputer.fit_transform(X)
        return X

    def transform(self, X):
        X = self.imputer.transform(X)
        return X 

    def save(self):

        with open(os.path.join('preprocessing', 'saved_components', 'imputers', 'imputer.pkl'), 'wb') as f:
            pickle.dump(self.imputer, f)

    def load(self):

        with open(os.path.join('preprocessing', 'saved_components', 'imputers', 'imputer.pkl'), 'rb') as f:
            self.imputer = pickle.load(f)