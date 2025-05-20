"""
Official Implementation of DeepSAD: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master 
"""

import json
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger
from torch.utils.data import TensorDataset, DataLoader
from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
import os
import torch
import torch.optim as optim
import numpy as np
import time 

logger = get_logger(__name__)


class DeepSAD(BaseModel):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, **kwargs):
        
        self.model_type = kwargs['model_type']
        self.modality = kwargs['modality']
        self.c = None 
        self.net = None  
        self.trainer = None

        self.ae_net = None  
        self.ae_trainer = None
        
        self.layers = kwargs['layers']
        self.learning_rate = kwargs['learning_rate']
        self.num_epochs = kwargs['num_epochs']
        self.lr_milestones = kwargs.get('lr_milestones', None)
        self.batch_size = kwargs['batch_size']
        self.weight_decay = kwargs['weight_decay']
        self.device  = kwargs['device']
        self.eta = kwargs['eta']
        self._preprocessor = self.init_preprocessor(**kwargs)
        self.input_d = kwargs.get('input_d', None)
        self.threshold = kwargs.get('threshold', None)
        self.pretrain = kwargs['pretrain']
        self.contamination = kwargs['contamination']

        self.trainer = DeepSADTrainer(**kwargs)
        
    @property
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor

    
    def fit(self, X, y):
        X = self._preprocessor.fit_transform(X)
        self._preprocessor.save()
        self.input_d = X.shape[1]
        self._construct_model()

        if self.pretrain:
            self.pretrain_model(X)
        self.net = self.trainer.train(X, y, self.net)
        self.c = self.trainer.c
        self.threshold = self.trainer.threshold
        

    def predict(self, X):
        self._preprocessor.load()
        X = self._preprocessor.transform(X)
        self.net.eval()
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.net(X)
            scores = torch.sum((outputs - self.c) ** 2, dim=1).cpu().numpy()
        return np.where(scores > self.threshold, 1, 0)

    def predict_proba(self, X):
        self._preprocessor.load()
        X = self._preprocessor.transform(X)
        self.net.eval()
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.net(X)
            scores = torch.sum((outputs - self.c) ** 2, dim=1).cpu().numpy()
        return scores
    

    def pretrain_model(self, X, lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda'):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        self._construct_ae()

        self.ae_trainer = AETrainer()
        
        self.ae_net = self.ae_trainer.train(X, self.ae_net)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save(self):
        """Save Deep SAD model to export_model."""

        path = os.path.join(self.model_dir_path, "deepsad.pth")
        logger.info(f"Saving model at {path} ...")
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if self.pretrain else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, path)
        logger.info("Model saved!")

    def load(self):
        """Load Deep SAD model from model_path."""
        self._construct_model()
        self._construct_ae()

        path = os.path.join(self.model_dir_path, "deepsad.pth")
        logger.info(f"Loading model from {path} ...")
        model_dict = torch.load(path)
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        self.net.to(self.device)

        
        if self.pretrain:
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])
            self.ae_net.to(self.device)

        
        
        logger.info("Model loaded!")

    def _construct_model(self):
        self.net = MLP(x_dim=self.input_d, h_dims=self.layers[:-1], rep_dim=self.layers[-1], bias=False)

    def _construct_ae(self):
        self.ae_net = MLP_Autoencoder(x_dim=self.input_d,  h_dims=self.layers[:-1], rep_dim=self.layers[-1], bias =False)

class DeepSADTrainer():

    def __init__(self, **kwargs):

        self.eta = kwargs['eta'] 
        self.lr = kwargs['learning_rate'] 
        self.num_epochs = kwargs['num_epochs']
        self.batch_size = kwargs['batch_size']
        self.weight_decay = kwargs['weight_decay']
        self.device = kwargs['device']
        self.contamination = kwargs['contamination']

        self.c = torch.tensor(kwargs['c'], device=self.device) if kwargs.get('c', None) is not None else None
        self.eps = 1e-6


    def train(self, X, y, net):

        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y.values).to(self.device)

        y[y==1] = -1

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        net.train()
        for epoch in range(self.num_epochs):

            # scheduler.step()
            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
 
            for data in loader:
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(targets == 0, dist, self.eta * ((dist + self.eps) ** targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            logger.info(f'| Epoch: {epoch + 1:03}/{self.num_epochs:03} '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
        
        net.eval()
        with torch.no_grad():
            self.decision_scores_ = torch.sum((net(X) - self.c) ** 2, dim=1)

        sorted_train, _ = torch.sort(self.decision_scores_, descending=False)
        threshold_index = (1 - self.contamination) * X.shape[0]
        threshold_index = int(threshold_index)
        self.threshold = sorted_train[threshold_index].item()
        return net

    def init_center_c(self, train_loader: DataLoader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c  

########### PRETRAINING ############
class AETrainer():

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 512, weight_decay: float = 1e-6, device: str = 'cuda'):
       
       self.optimizer_name   = optimizer_name
       self.lr = lr
       self.n_epochs = n_epochs
       self.lr_milestones = lr_milestones
       self.batch_size = batch_size
       self.weight_decay = weight_decay
       self.device = device


    def train(self, X, ae_net):

        X = torch.Tensor(X).to(self.device)

        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss(reduction='none')

        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            # scheduler.step()
            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for inputs in loader:
                
                inputs = inputs[0].to(self.device)

                optimizer.zero_grad()

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished pretraining.')

        return ae_net


class MLP_Decoder(nn.Module):

    def __init__(self, x_dim, h_dims, rep_dim, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)


class MLP_Autoencoder(nn.Module):

    def __init__(self, x_dim, h_dims, rep_dim, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MLP(x_dim, h_dims, rep_dim, bias)
        self.decoder = MLP_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class MLP(nn.Module):

    def __init__(self, x_dim, h_dims, rep_dim, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)

class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
    