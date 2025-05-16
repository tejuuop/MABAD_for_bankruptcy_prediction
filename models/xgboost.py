from xgboost import XGBClassifier
from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.tfidf_preprocessor import TFIDFpreprocessor
from preprocessing.pass_preproccessor import PASSpreprocessor
from preprocessing.concat_preprocessor import CONCATpreprocessor

from utils.logger import  get_logger
import os

logger = get_logger(__name__)

class XGBoost(BaseModel):
    def __init__(self, **kwargs):
        
        self.model_type = kwargs['model_type']  
        self.modality = kwargs['modality']
        
        self.tfidf_dim = kwargs['tfidf_dim']
        self.n_estimators = kwargs['n_estimators']
        self.max_depth = kwargs['max_depth'] 
        self.learning_rate = kwargs['learning_rate']
        self.objective = kwargs['objective']

        self.clf = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, objective=self.objective)
        
        if self.modality == 'mda_auditor':
            self._preprocessor = TFIDFpreprocessor(**kwargs)
        elif self.modality == 'financial':
            self._preprocessor = PASSpreprocessor()
        elif self.modality == 'multimodal':
            self._preprocessor = CONCATpreprocessor(**kwargs)

    @property
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor

    def fit(self, X, y):
        X = self._preprocessor.fit_transform(X)
        self._preprocessor.save()
        self.clf.fit(X, y)
    
    def predict(self, X):
        self._preprocessor.load()
        X=self._preprocessor.transform(X)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        self._preprocessor.load()
        X=self._preprocessor.transform(X)
        return self.clf.predict_proba(X)
    
    def save(self):
        path = os.path.join("models", "saved_models",f"{self.modality}",f"{self.model_type}.json")
        logger.info(f"Saving model at {path} ...")
        self.clf.save_model(path)
        logger.info("Model saved!")

    def load(self):
        path = os.path.join("models", "saved_models",f"{self.modality}",f"{self.model_type}.json")
        logger.info(f"Loading model from {path} ...")
        self.clf.load_model(path)
        logger.info("Model loaded!")