from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import numpy as np
from utils.logger import  get_logger
import os
from tqdm import tqdm

logger = get_logger(__name__)


class DevNet(BaseModel,nn.Module):

    def __init__(self, **kwargs):
        
        nn.Module.__init__(self)

        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']

        self.num_epochs = kwargs['num_epochs']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.device = kwargs['device']
        self.l = kwargs['l']
        self.a = kwargs['a']
        self.contamination = kwargs['contamination']
        self.layers = kwargs['layers']
        self.weight_decay = kwargs['weight_decay']
        self.relu = nn.ReLU()
        self._preprocessor = self.init_preprocessor(**kwargs)
        self.input_d = kwargs.get('input_d', None)
        self.threshold = kwargs.get('threshold', None)

    @property
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor
    
    def fit(self, X, y):
        
        X = self._preprocessor.fit_transform(X)
        self.input_d = X.shape[1]

        self._construct_model()
        self._initialize_weights()

        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y.values).to(self.device)

        positives = X[y == 1]
        negatives = X[y == 0] 

        optimizer = torch.optim.RMSprop(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) 

        negatives_dataset = TensorDataset(negatives)
        dataloader = DataLoader(
            negatives_dataset, batch_size=self.batch_size, shuffle=True)

        self.train()

        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            for neg_batch in dataloader:
                optimizer.zero_grad()

                positive_batch_sampled = torch.randint(
                    0, positives.shape[0], (self.batch_size,))

                pos_batch = positives[positive_batch_sampled].to(self.device)

                anomaly_normal_scores = torch.normal(
                    torch.zeros(self.l), torch.ones(self.l)*1)
                mean = anomaly_normal_scores.mean()
                std = anomaly_normal_scores.std()

                dev_x_pos = (self(pos_batch) - mean)/std
                pos_loss = torch.where(
                    self.a - dev_x_pos > 0, self.a - dev_x_pos, 0)

                dev_x_neg = (self(neg_batch[0]) - mean)/std
                neg_loss = torch.abs(dev_x_neg)

                loss = torch.mean(torch.cat([pos_loss, neg_loss]))

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            logger.info(f"Epoch: {epoch+1}/{self.num_epochs}, loss (sum): {epoch_loss}")

        self.eval()
        with torch.no_grad():
            self.decision_scores_ = self(X).squeeze(1)

        sorted_train, _ = torch.sort(self.decision_scores_, descending=False)
        threshold_index = (1 - self.contamination) * X.shape[0]
        threshold_index = int(threshold_index)
        self.threshold = sorted_train[threshold_index].item()

    def predict(self, X):
        X = self._preprocessor.transform(X)
        self.eval()
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            output = self(X).squeeze(1).cpu().numpy()
        return np.where(output > self.threshold, 1, 0)

    def predict_proba(self, X):
        X = self._preprocessor.transform(X)
        self.eval()
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            output = self(X).squeeze(1).cpu().numpy()
        return output

    def save(self):
        path = os.path.join(self.model_dir_path, "devnet.pth")
        logger.info(f"Saving model at {path} ...")
        torch.save(self.state_dict(), path)
        logger.info("Model saved!")

    def load(self):
        self._construct_model()
        path = os.path.join(self.model_dir_path, "devnet.pth")
        logger.info(f"Loading model from {path} ...")
        self.load_state_dict(torch.load(path))  
        logger.info("Model loaded!")

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    
    def forward(self, x):

        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i != self.mlp.__len__() - 1:
                x = self.relu(x)
        return x
    
    def _construct_model(self):
        layers = [self.input_d] + self.layers
        
        self.mlp = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.mlp.append(nn.Linear(layers[-1], 1))

        self.to(self.device)