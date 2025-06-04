import joblib
from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.pass_preproccessor import PASSpreprocessor
from tabpfn import TabPFNClassifier
from utils.logger import get_logger
import pickle
import os

logger = get_logger(__name__)

class TabPFN(BaseModel):

    def __init__(self, **kwargs):
        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']


        self.clf = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)

        self._preprocessor = PASSpreprocessor()
        
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
        return self.clf.predict_proba(X)[:,1]

    def save(self):
        path = os.path.join(self.model_dir_path,"tabpfn.pkl")
        logger.info(f"Saving model at {path} ...")
        
        # from tabpfn.model.loading import save_tabpfn_model
        # save_tabpfn_model(self.clf, path)
        with open(path, 'wb') as f:
            joblib.dump(self.clf, path)
            # pickle.dump(self.clf, f)
        logger.info("Model saved!")

    def load(self):
        path = os.path.join(self.model_dir_path,"tabpfn.pkl")
        logger.info(f"Loading model from {path} ...")
        with open(path, 'rb') as f:
            self.clf = joblib.load(f)
        # from tabpfn.model.loading import load_model
        #self.clf, _, _ = load_model(path=path, model_seed=42)
        # self.clf = TabPFNClassifier(model_path=path)
        logger.info("Model loaded!")

