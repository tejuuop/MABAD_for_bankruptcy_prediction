from preprocessing.base_preprocessor import BasePreprocessor
from sklearn.impute import SimpleImputer
import pickle
import os


class MeanImputer(BasePreprocessor):

    def __init__(self, strategy):
        if strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        else:
            self.imputer = SimpleImputer(
                strategy='constant', fill_value=int(strategy))

    def fit_transform(self, X):
        return self.imputer.fit_transform(X)

    def transform(self, X):
        return self.imputer.transform(X)

    def save(self):

        with open(os.path.join('preprocessing', 'saved_components', 'imputers', 'mean_imputer.pkl'), 'wb') as f:
            pickle.dump(self.imputer, f)

    def load(self):

        with open(os.path.join('preprocessing', 'saved_components', 'imputers', 'mean_imputer.pkl'), 'rb') as f:
            self.imputer = pickle.load(f)
