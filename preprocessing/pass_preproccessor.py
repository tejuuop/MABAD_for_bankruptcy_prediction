from preprocessing.base_preprocessor import BasePreprocessor

class PASSpreprocessor(BasePreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X
    
    def save(self, modality = None):
        pass
            
    def load(self, modality = None):
        pass     
        