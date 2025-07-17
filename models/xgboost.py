import numpy as np
import xgboost as xgb
from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
from utils.logger import get_logger
import scipy.sparse
import pandas as pd

logger = get_logger(__name__)


def focal_loss(alpha=0.25, gamma=2.0):
    def fl_obj(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # sigmoid

        # Clip predictions to prevent log(0) and division by 0
        preds = np.clip(preds, 1e-6, 1 - 1e-6)

        # Compute pt
        pt = preds * labels + (1 - preds) * (1 - labels)
        alpha_factor = alpha * labels + (1 - alpha) * (1 - labels)
        gamma_factor = (1 - pt) ** gamma

        # Gradient
        grad = alpha_factor * gamma_factor * (preds - labels)

        # Hessian
        hess = alpha_factor * gamma_factor * preds * (1 - preds) * (
            1 + gamma * (1 - pt) * (1 - 2 * preds)
        )

        return grad, hess
    return fl_obj




class XGBoost(BaseModel):
    def __init__(self, **kwargs):
        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']
        self.n_estimators = kwargs['n_estimators']
        self.max_depth = kwargs['max_depth']
        self.learning_rate = kwargs['learning_rate']
        self.alpha = kwargs.get('alpha', 0.25)
        self.gamma = kwargs.get('gamma', 2.0)
        self.threshold = kwargs.get("threshold", 0.5)

        self._preprocessor = self.init_preprocessor(**kwargs)

        self.clf = None  # will be xgb.Booster after training

    @property
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor

    def _prepare_input(self, X):
        """Concatenate and return NumPy array"""
        X_tab, X_mda, X_aud = X
        X = np.concatenate([X_tab, X_mda, X_aud], axis=1)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return X

    def fit(self, X, y):
        X_tab, X_mda, X_aud = self._preprocessor.fit_transform(X)
        X_np = self._prepare_input((X_tab, X_mda, X_aud))
        y_np = np.array(y)

        dtrain = xgb.DMatrix(X_np, label=y_np)
        
        num_positives = y.sum()
        num_negatives = y.shape[0] - num_positives
        pos_weight = num_negatives / num_positives
        
        params = {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "eval_metric": "logloss",
            "tree_method": "gpu_hist",
            "random_state": 42,
            "seed": 42,
            "objective": "binary:logistic",
            # "scale_pos_weight": pos_weight
        }

        self.clf = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            obj=focal_loss(alpha=self.alpha, gamma=self.gamma),
            verbose_eval=False
        )

    def predict(self, X):
        X_tab, X_mda, X_aud = self._preprocessor.transform(X)
        X_np = self._prepare_input((X_tab, X_mda, X_aud))
        dtest = xgb.DMatrix(X_np)
        preds = self.clf.predict(dtest)
        return (preds > self.threshold).astype(int)

    def predict_proba(self, X):
        X_tab, X_mda, X_aud = self._preprocessor.transform(X)
        X_np = self._prepare_input((X_tab, X_mda, X_aud))
        dtest = xgb.DMatrix(X_np)
        return self.clf.predict(dtest)

    def save(self):
        logger.info(f"Saving model at {self.model_path} ...")
        self.clf.save_model(self.model_path)
        logger.info("Model saved!")

    def load(self):
        logger.info(f"Loading model from {self.model_path} ...")
        self.clf = xgb.Booster()
        self.clf.load_model(self.model_path)
        logger.info("Model loaded!")
