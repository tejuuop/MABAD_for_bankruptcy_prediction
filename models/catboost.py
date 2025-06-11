from catboost import CatBoostClassifier
from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor


from utils.logger import get_logger

logger = get_logger(__name__)

class CatBoost(BaseModel):
    def __init__(self, **kwargs):
        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']

        self.n_iterations = kwargs['n_iterations']
        self.depth = kwargs['depth']
        self.learning_rate = kwargs['learning_rate']
        self.loss_function = kwargs['loss_function']  

        self._preprocessor = self.init_preprocessor(**kwargs)
        
        self.clf = CatBoostClassifier(
            iterations=self.n_iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            task_type='GPU',  
            logging_level='Silent',   
            allow_writing_files=False 
        )

    @property
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor


    def fit(self, X, y):
        X = self._preprocessor.fit_transform(X)
        self.clf.fit(X, y)

    def predict(self, X):
        X = self._preprocessor.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = self._preprocessor.transform(X)
        return self.clf.predict_proba(X)[:,1]   

    def save(self):
        logger.info(f"Saving model at {self.model_path} ...")
        self.clf.save_model(self.model_path)
        logger.info("Model saved!")

    def load(self):
        logger.info(f"Loading model from {self.model_path} ...")
        self.clf.load_model(self.model_path)
        logger.info("Model loaded!")
