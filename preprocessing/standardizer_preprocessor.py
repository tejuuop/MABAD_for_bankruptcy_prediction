from preprocessing.base_preprocessor import BasePreprocessor
from sklearn.preprocessing import StandardScaler
import pickle
import os


class Standardizer(BasePreprocessor):

    def __init__(self):
        self.standardizer = StandardScaler()

    def fit_transform(self, X):
        return self.standardizer.fit_transform(X)
    

    def transform(self, X):
        return self.standardizer.transform(X)
    

    def save(self):

        with open(os.path.join('preprocessing','saved_components','standardizer','standardizer.pkl'), 'wb') as f:
            pickle.dump(self.standardizer, f)


    def load(self):
        with open(os.path.join('preprocessing','saved_components','standardizer','standardizer.pkl'), 'rb') as f:
            self.standardizer = pickle.load(f)
