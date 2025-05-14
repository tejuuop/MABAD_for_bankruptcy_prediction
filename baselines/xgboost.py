from xgboost import XGBClassifier
from models.base_model import BaseModel

class XGBoost(BaseModel):
    def __init__(self, **kwargs):
        self.clf = XGBClassifier(n_estimators=kwargs['n_estimators'], max_depth=kwargs['max_depth'], learning_rate=kwargs['learning_rate'], objective=kwargs['objective'])
        

    def fit(self, X, y):
        return self.clf.fit(y)
    
    def predict(self, X):
        return self.clf.predict()
    
    def predict_proba(self, X):
        return self.clf.predict_proba()