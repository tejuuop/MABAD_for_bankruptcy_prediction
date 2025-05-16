from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score
import numpy as np
from config import VAL_METRIC_TO_OPTIMIZE
import pandas as pd 
from utils.logger import  get_logger
from preprocessing.base_preprocessor import BasePreprocessor

logger = get_logger(__name__)

class BaseModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
    
    @property
    @abstractmethod
    def preprocessor(self) -> BasePreprocessor:
        pass

    @property
    def current_validation_performance(self):

        metrics = {"auc" : self.auc,
                    "ap" : self.ap ,
                    "fbeta" : self.fbeta,
                    "recall_at_precision" : self.recall_at_precision}
        
        return metrics[VAL_METRIC_TO_OPTIMIZE]


    def evaluate(self, X, y, dataset):

        preds = self.predict(X)
        scores = self.predict_proba(X)[:,1]    

        self.auc = roc_auc_score(y_score=scores, y_true=y)
        self.ap = average_precision_score(y_score=scores, y_true=y)
        self.fbeta = fbeta_score(y_pred=preds, y_true=y , beta=1.5, average='binary') # we focus on the bankruptcy case with more interest on the recall metric. threshold is 0.5

        #recall_at_precision at threshold τ=number of true positives
        num_of_positives = y.sum() 
        sorted_indices = np.argsort(scores)[::-1]
        found_labels = y.reset_index(drop=True).loc[sorted_indices]
        self.recall_at_precision = found_labels[:num_of_positives].sum() / num_of_positives
        
        self.auc = round(self.auc, 2)
        self.ap = round(self.ap,2)
        self.fbeta = round(self.fbeta, 2)
        self.recall_at_precision = round(self.recall_at_precision,2)

        results = {
            "model": self.model_type,
            "auc": self.auc,
            "ap": self.ap,
            "fbeta": self.fbeta,
            "recall_at_precision": self.recall_at_precision
        }

        if dataset == 'train':
            logger.info(f"Training set results: {results}")
        elif dataset == 'val':
            logger.info(f"Validation set results: {results}")
        else:
            logger.info(f"Test set results: {results}")

    def save_hyperparameters(self):
        #save only hyperparameter class attributes
        hyperparameters = {k:v for k,v in self.__dict__.items() if isinstance(v, (int, float, str, bool, type(None)))} 
        
        timestamp = datetime.now().isoformat()
        new_record = {timestamp : hyperparameters}
        json_path = os.path.join('hyperparameters',f"{self.modality}",f"{self.model_type}.json")

        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                existing_hyperparameters = json.load(file)

            new_record.update(existing_hyperparameters)

        with open(json_path, 'w') as file:
            json.dump(new_record, file, indent=4)

    def previous_best_validation_performance(self):

        json_path = os.path.join('hyperparameters', f"{self.modality}", f"{self.model_type}.json")

        if not os.path.exists(json_path):
            logger.info(f"No previous hyperparameter file found for model: {self.model_type} and modality: {self.modality}")
            return -1

        with open(json_path, 'r') as file:
            prev_hyperparameters = json.load(file)

        
        best_metric = -1
        best_hyperparams = {}
        for _, values in prev_hyperparameters.items():
            metric_value = values[VAL_METRIC_TO_OPTIMIZE]
            if metric_value > best_metric:
                best_metric = metric_value
                for hyperparameter, hyper_value in values.items():
                    if hyperparameter in self.__dict__:
                        best_hyperparams[hyperparameter] = hyper_value

        return best_metric
        
    def save_results(self):
        """
        Save the test set's results
        """

        csv_path = os.path.join('results',f"{self.modality}.csv")

        results = pd.DataFrame([{
            "model": f"{self.model_type}",
            "auc": f"{self.auc:.2f}",
            "ap": f"{self.ap:.2f}",
            "fbeta": f"{self.fbeta:.2f}",
            "recall_at_precision": f"{self.recall_at_precision:.2f}"
        }])

        if os.path.exists(csv_path):
            previous_results = pd.read_csv(csv_path)
            previous_results = previous_results[previous_results.model != self.model_type] #keep the latest one
            results = pd.concat([results, previous_results], axis = 0)

        results.to_csv(csv_path, index=False)