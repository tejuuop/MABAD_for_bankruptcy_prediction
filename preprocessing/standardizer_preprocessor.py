from preprocessing.base_preprocessor import BasePreprocessor
from sklearn.preprocessing import StandardScaler
import pickle
import os
import scipy.sparse


class StandardizerPreprocessor(BasePreprocessor):

    def __init__(self, source):
        self.standardizer = StandardScaler()
        self.source = source

    def fit_transform(self, X):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return self.standardizer.fit_transform(X)

    def transform(self, X):
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return self.standardizer.transform(X)

    def save(self):

        with open(os.path.join('preprocessing','saved_components','standardizer',f'standardizer_{self.source}.pkl'), 'wb') as f:
                        pickle.dump(self.standardizer, f)


    def load(self):

        with open(os.path.join('preprocessing','saved_components','standardizer',f'standardizer_{self.source}.pkl'), 'rb') as f:
                        self.standardizer = pickle.load(f)
