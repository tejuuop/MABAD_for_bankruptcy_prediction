from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

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

    def run(self, data):
        self._split(data)

    def _split(self, data):
        y = data['label'] 
        X = data.drop(columns=['label'])

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.666, stratify=y_temp, random_state=42
        )


        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        print(f"Positives - Train: {y_train.sum()}, Validation: {y_val.sum()}, Test: {y_test.sum()}")

