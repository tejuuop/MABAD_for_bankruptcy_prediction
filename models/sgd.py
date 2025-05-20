from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
from sklearn.linear_model import SGDClassifier
from utils.logger import get_logger
import pickle
import os

logger = get_logger(__name__)

class SGD(BaseModel):

    def __init__(self, **kwargs):
        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']

        self.loss = kwargs['loss']
        self.penalty = kwargs['penalty']
        self.max_iter = kwargs['max_iter']
        
        self.clf = SGDClassifier(loss=self.loss, penalty=self.penalty, max_iter=self.max_iter)
        self._preprocessor = self.init_preprocessor(**kwargs)
        
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor
    
    def fit(self, X, y):
        X = self._preprocessor.fit_transform(X)
        self._preprocessor.save()
        self.clf.fit(X, y)

    def predict(self, X):
        self._preprocessor.load()
        X = self._preprocessor.transform(X)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        self._preprocessor.load()
        X = self._preprocessor.transform(X)
        return self.clf.decision_function(X)

    def save(self):
        path = os.path.join(self.model_dir_path,"sgd.pkl")
        logger.info(f"Saving model at {path} ...")
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
        logger.info("Model saved!")

    def load(self):
        path = os.path.join(self.model_dir_path,"sgd.pkl")
        logger.info(f"Loading model from {path} ...")
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
        logger.info("Model loaded!")

