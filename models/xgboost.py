from xgboost import XGBClassifier
from models.base_model import BaseModel

class XGBoost(BaseModel):
    def __init__(self, **kwargs):
        
        self.model_type = kwargs['model_type']  
        self.modality = kwargs['modality']
        
        self.n_estimators = kwargs['n_estimators']
        self.max_depth = kwargs['max_depth'] 
        self.learning_rate = kwargs['learning_rate']
        self.objective = kwargs['objective']

        self.clf = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, objective=self.objective)

    def fit(self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    
