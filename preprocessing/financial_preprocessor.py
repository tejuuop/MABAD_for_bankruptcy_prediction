from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.pass_preproccessor import PASSpreprocessor
from preprocessing.dense_cols_preprocessor import DenseColsPreprocessor
from preprocessing.imputer_preprocessor import ImputerPreprocessor
from preprocessing.standardizer_preprocessor import StandardizerPreprocessor

class FinancialPreprocessor(BasePreprocessor):

    def __init__(self, **kwargs):
        
        self.dense_cols_preprocessor = DenseColsPreprocessor(**kwargs) if kwargs['num_dense_cols'] else PASSpreprocessor()
        self.imputer_prepocessor = ImputerPreprocessor(**kwargs) if kwargs['imputer'] else PASSpreprocessor()
        self.standardizer_preprocessor = StandardizerPreprocessor(**kwargs) if  kwargs['standardize'] else PASSpreprocessor()
        self.preprocessors = [self.dense_cols_preprocessor, self.imputer_prepocessor, self.standardizer_preprocessor]

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
    