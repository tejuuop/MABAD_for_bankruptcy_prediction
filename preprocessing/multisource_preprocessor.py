from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.financial_preprocessor import FinancialPreprocessor
from preprocessing.tfidf_preprocessor import TFIDFpreprocessor
from preprocessing.standardizer_preprocessor import StandardizerPreprocessor
from preprocessing.pass_preproccessor import PASSpreprocessor
import numpy as np
import scipy.sparse as sp
from utils.logger import get_logger

logger = get_logger(__name__)

class MultisourcePreprocessor(BasePreprocessor):
    """
    Return text + tabular
    """

    def __init__(self, **kwargs):

        self.load_tfidfs = kwargs['load_tfidfs']
        logger.info(f"Initializing multisource preprocessor [load_tfidfs = {self.load_tfidfs}]...")
        self.mda_auditor_preprocessor = TFIDFpreprocessor(**kwargs) if not self.load_tfidfs else PASSpreprocessor()
        self.financial_preprocessor = FinancialPreprocessor(**kwargs) # Dense -> Impute -> Standardization OR (change manually) Dense -> standardize ZERO impute

        self._standardizer_preprocessor_mda = StandardizerPreprocessor(source='mda') if kwargs['standardize'] else PASSpreprocessor()
        self._standardizer_preprocessor_auditor = StandardizerPreprocessor(source='auditor') if kwargs['standardize'] else PASSpreprocessor()
        # self._standardizer_preprocessor_mda = PASSpreprocessor()
        # self._standardizer_preprocessor_auditor = PASSpreprocessor()

    
    def fit_transform(self, X):
        if not self.load_tfidfs:
            X_financial, X_mda_auditor = X   
            logger.info("[Fit Transform] Extracting tfidfs...")
            X_mda, X_auditor = self.mda_auditor_preprocessor.fit_transform(X_mda_auditor)
            logger.info("Done!")
        else:
            X_financial, X_mda, X_auditor = X

        X_financial = self.financial_preprocessor.fit_transform(X_financial)    
        X_mda = self._standardizer_preprocessor_mda.fit_transform(X_mda)
        X_auditor = self._standardizer_preprocessor_auditor.fit_transform(X_auditor)

        return self._to_numpy(X_financial, X_mda, X_auditor)

    def transform(self, X):
        if not self.load_tfidfs:
            X_financial, X_mda_auditor = X   
            logger.info("[Transform] Extracting tfidfs...")
            X_mda, X_auditor = self.mda_auditor_preprocessor.transform(X_mda_auditor)
            logger.info("Done!")
        else:
            X_financial, X_mda, X_auditor = X

        X_financial = self.financial_preprocessor.transform(X_financial)    
        X_mda = self._standardizer_preprocessor_mda.transform(X_mda)
        X_auditor = self._standardizer_preprocessor_auditor.transform(X_auditor)

        return self._to_numpy(X_financial, X_mda, X_auditor)


    def _to_numpy(self, X_financial, X_mda, X_auditor):
        """
        Convert any incoming object (DataFrame, sparse matrix, list, ndarray)
        to a dense NumPy array (dtype=float32) so that downstream code
        always receives three dense arrays.
        """
        def _as_dense_array(obj):
            # Pandas DataFrame / Series → ndarray
            if hasattr(obj, "to_numpy"):
                obj = obj.to_numpy()

            # SciPy sparse → dense
            if sp.issparse(obj):
                obj = obj.toarray()

            # Anything else → ndarray
            return np.asarray(obj, dtype=np.float32)

        return (
            _as_dense_array(X_financial),
            _as_dense_array(X_mda),
            _as_dense_array(X_auditor),
        )


    def save(self):
        self.mda_auditor_preprocessor.save()
        self.financial_preprocessor.save()
        self._standardizer_preprocessor_mda.save()
        self._standardizer_preprocessor_auditor.save()

    def load(self):
        self.mda_auditor_preprocessor.load()
        self.financial_preprocessor.load()
        self._standardizer_preprocessor_mda.load()
        self._standardizer_preprocessor_auditor.load()
