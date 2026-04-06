from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.financial_preprocessor import FinancialPreprocessor
from preprocessing.tfidf_preprocessor import TFIDFpreprocessor
from preprocessing.finbert_preprocessor import FinBERTPreprocessor # <-- ADDED IMPORT
from preprocessing.standardizer_preprocessor import StandardizerPreprocessor
from preprocessing.pass_preproccessor import PASSpreprocessor
import numpy as np
import scipy.sparse as sp
from utils.logger import get_logger

logger = get_logger(__name__)

class MultisourcePreprocessor(BasePreprocessor):
    """
    Return text + tabular with Smart Routing for TF-IDF vs FinBERT
    """

    def __init__(self, **kwargs):
        self.load_tfidfs = kwargs['load_tfidfs']
        self.tfidf_dim = kwargs.get('tfidf_dim', 1000) # Check the command line arg
        
        logger.info(f"Initializing multisource preprocessor [load_tfidfs = {self.load_tfidfs}, dim = {self.tfidf_dim}]...")
        
        # === THE SMART SWITCH ===
        if not self.load_tfidfs:
            if self.tfidf_dim == 768:
                logger.info("🤖 SMART SWITCH: 768 dimensions detected. Routing text to FinBERT.")
                # Pass the device from kwargs, defaulting to cuda:0 just in case
                self.mda_auditor_preprocessor = FinBERTPreprocessor(device=kwargs.get('device', 'cuda:0'))
            else:
                logger.info(f"📊 SMART SWITCH: {self.tfidf_dim} dimensions detected. Routing text to TF-IDF.")
                self.mda_auditor_preprocessor = TFIDFpreprocessor(**kwargs)
        else:
            self.mda_auditor_preprocessor = PASSpreprocessor()
        # ========================

        self.financial_preprocessor = FinancialPreprocessor(**kwargs)

        self._standardizer_preprocessor_mda = StandardizerPreprocessor(source='mda') if kwargs['standardize'] else PASSpreprocessor()
        self._standardizer_preprocessor_auditor = StandardizerPreprocessor(source='auditor') if kwargs['standardize'] else PASSpreprocessor()

    def fit_transform(self, X):
        if not self.load_tfidfs:
            X_financial, X_mda_auditor = X   
            logger.info("[Fit Transform] Extracting NLP embeddings...")
            
            # Safe call: FinBERT doesn't need 'fitting' like TF-IDF does, so we fall back to 'transform' safely
            if hasattr(self.mda_auditor_preprocessor, 'fit_transform'):
                X_mda, X_auditor = self.mda_auditor_preprocessor.fit_transform(X_mda_auditor)
            else:
                X_mda, X_auditor = self.mda_auditor_preprocessor.transform(X_mda_auditor)
                
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
            logger.info("[Transform] Extracting NLP embeddings...")
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
        Convert any incoming object to a dense NumPy array.
        """
        def _as_dense_array(obj):
            if hasattr(obj, "to_numpy"):
                obj = obj.to_numpy()
            if sp.issparse(obj):
                obj = obj.toarray()
            return np.asarray(obj, dtype=np.float32)

        return (
            _as_dense_array(X_financial),
            _as_dense_array(X_mda),
            _as_dense_array(X_auditor),
        )

    def save(self):
        # Safe save in case our custom FinBERT doesn't have a save method
        if hasattr(self.mda_auditor_preprocessor, 'save'):
            self.mda_auditor_preprocessor.save()
        self.financial_preprocessor.save()
        self._standardizer_preprocessor_mda.save()
        self._standardizer_preprocessor_auditor.save()

    def load(self):
        # Safe load in case our custom FinBERT doesn't have a load method
        if hasattr(self.mda_auditor_preprocessor, 'load'):
            self.mda_auditor_preprocessor.load()
        self.financial_preprocessor.load()
        self._standardizer_preprocessor_mda.load()
        self._standardizer_preprocessor_auditor.load()