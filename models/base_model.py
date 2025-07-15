from abc import ABC, abstractmethod
import json
import os
from datetime import datetime
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score
import numpy as np
from config import VAL_METRIC_TO_OPTIMIZE
import pandas as pd
from utils.logger import get_logger
from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.financial_preprocessor import FinancialPreprocessor
from preprocessing.tfidf_preprocessor import TFIDFpreprocessor
from preprocessing.multisource_preprocessor import MultisourcePreprocessor
from preprocessing.embeddings_preprocessor import EmbeddingsPreprocessor


logger = get_logger(__name__)


class BaseModel(ABC):

    @property
    def model_dir_path(self) -> str:
        path = os.path.join("models", "saved_models", self.modality, self.model_type)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir_path, "model.json")

    @property
    def best_hyperparameters_path(self) -> str:
        return os.path.join(self.model_dir_path, "hyperparameters.json")

    @property
    def current_validation_performance(self):

        metrics = {
            "auc": self.auc,
            "ap": self.ap,
            "fbeta": self.fbeta,
            "recall_at_precision": self.recall_at_precision,
        }

        return metrics[VAL_METRIC_TO_OPTIMIZE]

    @property
    @abstractmethod
    def preprocessor(self) -> BasePreprocessor:
        pass

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

    @staticmethod
    def load_hyperparameter_dict(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    @classmethod
    def from_hyperparams(cls, hyperparameter_path):
        hyperparameters = cls.load_hyperparameter_dict(hyperparameter_path)
        logger.info(f"Loaded hyperparameters: {hyperparameters}")

        return cls(**hyperparameters)

    def evaluate(self, X, y, dataset):

        preds = self.predict(X)
        scores = self.predict_proba(X)

        self.auc = roc_auc_score(y_score=scores, y_true=y)
        self.ap = average_precision_score(y_score=scores, y_true=y)
        # we focus on the bankruptcy case with more interest on the recall metric. threshold is 0.5
        self.fbeta = fbeta_score(y_pred=preds, y_true=y, beta=1.5, average="binary")

        # recall_at_precision at threshold τ=number of true positives
        num_of_positives = y.sum()
        sorted_indices = np.argsort(scores)[::-1]
        found_labels = y.reset_index(drop=True).loc[sorted_indices]
        self.recall_at_precision = (
            found_labels[:num_of_positives].sum() / num_of_positives
        )

        self.auc = round(self.auc, 2)
        self.ap = round(self.ap, 2)
        self.fbeta = round(self.fbeta, 2)
        self.recall_at_precision = round(self.recall_at_precision, 2)

        results = {
            "model": self.model_type,
            "auc": self.auc,
            "ap": self.ap,
            "fbeta": self.fbeta,
            "recall_at_precision": self.recall_at_precision,
        }

        if dataset == "train":
            logger.info(f"Training set results: {results}")
        elif dataset == "val":
            logger.info(f"Validation set results: {results}")
        else:
            logger.info(f"Test set results: {results}")

    def save_hyperparameters(self, hyperparameters, qualify_for_testing=False):
        
        extra_hyperparameters = {
            k: v
            for k, v in self.__dict__.items()
            if isinstance(
                v, (int, float, str, bool, type(None), tuple, list, pd.Series)
            )
        }

        hyperparameters = {
            k: v
            for k, v in hyperparameters.items()
            if isinstance(
                v, (int, float, str, bool, type(None), tuple, list, pd.Series)
            )
        }
        # hyperparameters["ap"] = self.ap
        # hyperparameters["auc"] = self.auc
        hyperparameters.update(extra_hyperparameters)

        timestamp = datetime.now().isoformat()
        new_record = {timestamp: hyperparameters}
        json_path = os.path.join(
            "hyperparameters", f"{self.modality}", f"{self.model_type}.json"
        )

        if os.path.exists(json_path):
            with open(json_path, "r") as file:
                existing_hyperparameters = json.load(file)

            new_record.update(existing_hyperparameters)

        with open(json_path, "w") as file:
            json.dump(new_record, file, indent=4)

        if qualify_for_testing:
            with open(self.best_hyperparameters_path, "w") as file:
                json.dump(hyperparameters, file, indent=4)

    def previous_best_validation_performance(self):

        json_path = os.path.join(
            "hyperparameters", f"{self.modality}", f"{self.model_type}.json"
        )

        if not os.path.exists(json_path):
            logger.info(
                f"No previous hyperparameter file found for model: {self.model_type} and modality: {self.modality}"
            )
            return -1, {}

        with open(json_path, "r") as file:
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

        return best_metric, best_hyperparams

    def save_results(self, model_name):
        """
        Save the test set's results
        """
        if self.modality == 'multisource':
            self.modality = 'multimodal'
            
        csv_path = os.path.join("results", f"{self.modality}.csv")

        results = pd.DataFrame(
            [
                {
                    "model": f"{model_name}",
                    "auc": f"{self.auc:.2f}",
                    "ap": f"{self.ap:.2f}",
                    "fbeta": f"{self.fbeta:.2f}",
                    "recall_at_precision": f"{self.recall_at_precision:.2f}",
                }
            ]
        )

        if os.path.exists(csv_path):
            previous_results = pd.read_csv(csv_path)
            # keep the latest one
            previous_results = previous_results[previous_results.model != model_name]
            results = pd.concat([results, previous_results], axis=0)

        results.to_csv(csv_path, index=False)

    def init_preprocessor(self, **kwargs):

        if kwargs["modality"] == "mda_auditor":
            return TFIDFpreprocessor(**kwargs)
        elif kwargs["modality"] == "financial":
            return FinancialPreprocessor(**kwargs)
        elif kwargs["modality"] == "multisource":
            return MultisourcePreprocessor(**kwargs)


    def get_decision_threshold(self, X, y):
        logger.info("Computing best threshold on validation set...")
        scores = self.predict_proba(X)
        lower_bound = np.min(scores)
        upper_bound = np.max(scores)

        thresholds = np.linspace(start=lower_bound, stop=upper_bound, num=10000)
        best_threshold = -1
        best_fbeta = -1

        for threshold in thresholds:
            preds = np.where(scores > threshold, 1, 0)
            fbeta = fbeta_score(y_true=y, y_pred=preds, beta=1.5, average="binary")
            if fbeta > best_fbeta:
                best_fbeta = fbeta  # <-- missing update in your code
                best_threshold = threshold

        logger.info(f"Best threshold computed on validation set is {best_threshold}, fbeta score: {best_fbeta}")
        self.threshold = best_threshold

