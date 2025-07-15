from abc import ABC, abstractmethod
from utils.logger import  get_logger

logger = get_logger(__name__)

class BasePreprocessor(ABC):

    @abstractmethod
    def fit_transform(self, X):
        pass
    
    @abstractmethod
    def transform(self, X):
        pass
    
    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    