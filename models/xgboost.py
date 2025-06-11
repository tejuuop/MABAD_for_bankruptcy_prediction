from xgboost import XGBClassifier
from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
import cupy as cp
from utils.logger import  get_logger


logger = get_logger(__name__)

class XGBoost(BaseModel):
    def __init__(self, **kwargs):

        self.model_type = kwargs['model_type']  
        self.modality = kwargs['modality']
        self.n_estimators = kwargs['n_estimators']
        self.max_depth = kwargs['max_depth'] 
        self.learning_rate = kwargs['learning_rate']
        self.objective = kwargs['objective']

        self._preprocessor = self.init_preprocessor(**kwargs)
        self.clf = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, objective=self.objective, device="cuda")

    @property
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor
    

    def fit(self, X, y):
        X = self._preprocessor.fit_transform(X)
        logger.info(f"[Training] After preprocessing X shape is {X.shape}")
        X = cp.array(X)  
        y = cp.array(y)
        self.clf.fit(X, y)
    
    def predict(self, X):
        X=self._preprocessor.transform(X)
        logger.info(f"[Training] After preprocessing X shape is {X.shape}")
        X = cp.array(X) 
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        X=self._preprocessor.transform(X)
        logger.info(f"[Training] After preprocessing X shape is {X.shape}")
        X = cp.array(X) 
        return self.clf.predict_proba(X)[:,1]   
    
    def save(self):
        logger.info(f"Saving model at {self.model_path} ...")
        self.clf.save_model(self.model_path)
        logger.info("Model saved!")

    def load(self):
        logger.info(f"Loading model from {self.model_path} ...")
        self.clf.load_model(self.model_path)
        logger.info("Model loaded!")