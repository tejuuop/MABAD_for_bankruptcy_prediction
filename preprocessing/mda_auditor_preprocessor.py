from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.tfidf_preprocessor import TFIDFpreprocessor

class MDAauditorPreprocessor(BasePreprocessor):

    def __init__(self, **kwargs):
        
        self.tfidf_preprocessor = TFIDFpreprocessor(**kwargs) if kwargs['tfidf_dim'] else None
        self.preprocessors = [self.tfidf_preprocessor]

    def fit_transform(self, X):

        for p in self.preprocessors:
            X = p.fit_transform(X)
        return X
    
    
    def transform(self, X):
        for p in self.preprocessors:
            X = p.transform(X)
        return X
    
    
    
    def save(self):
        for p in self.preprocessors:
            p.save()   

    
    def load(self):
        for p in self.preprocessors:
            p.load()
