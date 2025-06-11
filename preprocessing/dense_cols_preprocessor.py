from preprocessing.base_preprocessor import BasePreprocessor
import pickle
import os 
class DenseColsPreprocessor(BasePreprocessor):

    def __init__(self, **kwargs):
        self.num_dense_cols = kwargs['num_dense_cols']
        self.dense_cols = []

    def fit_transform(self, X):
        self.dense_cols = list(X.notna().sum(axis=0).sort_values(ascending=False)[:self.num_dense_cols].index)
        return X.loc[:,self.dense_cols]
        

    def transform(self, X):
        return X.loc[:,self.dense_cols]

    def save(self):
        with open(os.path.join('preprocessing', 'saved_components', 'feature_selection', 'dense_cols.pkl'), "wb") as fp:   
            pickle.dump(self.dense_cols, fp)


    def load(self):
        with open(os.path.join('preprocessing', 'saved_components', 'feature_selection', 'dense_cols.pkl'), "rb") as fp:   
            self.dense_cols = pickle.load(fp)