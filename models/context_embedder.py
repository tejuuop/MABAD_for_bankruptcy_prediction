from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.pass_preproccessor import PASSpreprocessor
from utils.logger import get_logger
import pickle
import os
import pandas as pd
import torch.nn as nn 
import torch
import numpy as np
from tqdm import tqdm

logger = get_logger(__name__)

class CE(BaseModel, nn.Module):

    def __init__(self, **kwargs):

        nn.Module.__init__(self)

        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']
        self.batch_size = kwargs['batch_size']
        self.num_epochs = kwargs['num_epochs']

        self.feats_value_layers = kwargs['feats_value_layers']
        self.feats_context_layers = kwargs['feats_context_layers']
        self.classifier_hidden_dim = kwargs['classifier_hidden_dim']

        
        self.learning_rate = kwargs['learning_rate']
        self.weight_decay = kwargs['weight_decay']

        self._preprocessor = self.init_preprocessor(**kwargs)
        self.tag_description_embeddings_csv =  pd.read_csv('tag_description_embeddings.csv').set_index('tag')

        assert self.feats_value_layers[-1] == self.feats_context_layers[-1]

        self.feats_value_projection = None
        self.feats_context_projection =  None 

        self.classifier = nn.Sequential(
            nn.Linear(self.feats_context_layers[-1], self.classifier_hidden_dim),
            # nn.Linear(56, self.classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.classifier_hidden_dim, 1)
        )
        
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor
    
    def fit(self, X, y):

        self.input_d = X.shape[1]

        columns_ordered = X.columns

        self.C = torch.from_numpy(self.tag_description_embeddings_csv.loc[columns_ordered].iloc[:,1:].values).float()
        self.I = torch.from_numpy(~np.isnan(X.values)).int().float()
        print(self.I.shape)
        print(self.C.shape)
        self.context_d = self.C.shape[1]
        self._construct_feats_value_projection()
        self._construct_feats_context_projection()

        print(self)

        

        X = self._preprocessor.fit_transform(X)
        self._preprocessor.save()
       
        indices = torch.arange(len(X))  
        print(indices.shape)
        print(X.shape)

        trainset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y.values), indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) 

        self.train()

        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            for batch, labels, idx in trainloader:

                optimizer.zero_grad()

                outputs = self(batch, idx)

                loss = loss_fn(outputs.float(), labels.unsqueeze(-1).float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            logger.info(f"Epoch: {epoch+1}/{self.num_epochs}, loss (sum): {epoch_loss}")

    def predict(self, X):
        self._preprocessor.load()
        X = self._preprocessor.transform(X)
        outputs = self.predict_proba(X)
        preds = (outputs > 0.5).astype(int)
        return pd.Series(preds)
    
    def predict_proba(self, X):
        self._preprocessor.load()
        X = self._preprocessor.transform(X)
        X=torch.from_numpy(X).float()
        return self(X, torch.arange(len(X))  ).detach().numpy().flatten()

    def save(self):
        pass

    def load(self):
        pass


    def _construct_feats_value_projection(self):
        layers = [self.input_d] + self.feats_value_layers
        modules = []

        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2: 
                modules.append(nn.ReLU())

        self.feats_value_projection = nn.Sequential(*modules)

    def _construct_feats_context_projection(self):
        layers = [self.context_d] + self.feats_context_layers
        modules = []

        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.ReLU())

        self.feats_context_projection = nn.Sequential(*modules)


    def forward(self, X, indices):

        contextualized = self.feats_value_projection(X)
        # context = self.I[indices] @ self.C
        # context = self.feats_context_projection(context)
        # contextualized = torch.cat((X, context), dim=-1)
        return self.classifier(contextualized)
