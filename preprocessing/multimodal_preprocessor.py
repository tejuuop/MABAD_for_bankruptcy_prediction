from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.financial_preprocessor import FinancialPreprocessor
from preprocessing.mda_auditor_preprocessor import MDAauditorPreprocessor
from preprocessing.standardizer_preprocessor import StandardizerPreprocessor
from preprocessing.pass_preproccessor import PASSpreprocessor
import numpy as np


class MultimodalPreprocessor(BasePreprocessor):

    def __init__(self, **kwargs):

        self.mda_auditor_preprocessor = MDAauditorPreprocessor(**kwargs)
        self._standardizer_preprocessor = StandardizerPreprocessor(
            **kwargs) if kwargs['standardize'] else PASSpreprocessor()

        if kwargs['standardize'] == True:
            # Dont standardize financial feature in isolation
            kwargs['standardize'] = False

        self.financial_preprocessor = FinancialPreprocessor(**kwargs)

    def fit_transform(self, X):
        X_financial, X_mda_auditor = X
        X_financial = self.financial_preprocessor.fit_transform(X_financial)
        X_mda_auditor = self.mda_auditor_preprocessor.fit_transform(
            X_mda_auditor)
        X = np.concatenate([X_financial, X_mda_auditor], axis=1)
        X = self._standardizer_preprocessor.fit_transform(X)
        return X

    def transform(self, X):
        X_financial, X_mda_auditor = X
        X_financial = self.financial_preprocessor.transform(X_financial)
        X_mda_auditor = self.mda_auditor_preprocessor.transform(X_mda_auditor)
        X = np.concatenate([X_financial, X_mda_auditor], axis=1)
        X = self._standardizer_preprocessor.transform(X)
        return X

    def save(self):
        self.mda_auditor_preprocessor.save()
        self.financial_preprocessor.save()
        self._standardizer_preprocessor.save()

    def load(self):
        self.mda_auditor_preprocessor.load()
        self.financial_preprocessor.load()
        self._standardizer_preprocessor.load()
