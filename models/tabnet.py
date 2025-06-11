from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
from pytorch_tabnet.tab_model import TabNetClassifier
from utils.logger import get_logger
import pickle
import os

logger = get_logger(__name__)

class TabNet(BaseModel):

    def __init__(self, **kwargs):
        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']
        self.num_epochs = kwargs['num_epochs']
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['learning_rate']

        self.clf = TabNetClassifier(optimizer_params={'lr':self.lr})
        print(self.clf)
        self._preprocessor = self.init_preprocessor(**kwargs)
        
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor
    
    def fit(self, X, y):
        X = self._preprocessor.fit_transform(X.values)
        self.clf.fit(X, y,max_epochs=self.num_epochs,batch_size = self.batch_size, patience=5)

    def predict(self, X):
        X = self._preprocessor.transform(X.values)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        X = self._preprocessor.transform(X.values)
        return self.clf.predict_proba(X)[:,1]

    def save(self):
        path = os.path.join(self.model_dir_path)
        logger.info(f"Saving model at {path} ...")
        self.clf.save_model(os.path.join(path,"TabNet"))
        logger.info("Model saved!")
        

    def load(self):
        path = os.path.join(self.model_dir_path,"TabNet.zip")
        logger.info(f"Loading model from {path} ...")
        self.clf.load_model(path)
        logger.info("Model loaded!")
        

