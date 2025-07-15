from models.base_model import BaseModel
from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.multisource_preprocessor import MultisourcePreprocessor
from utils.logger import get_logger                                       
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = get_logger(__name__)


class MLP(BaseModel):
    """Multi-layer-perceptron wrapper that follows the BaseModel API."""
    def __init__(self, **kwargs):
       
        # ── bookkeeping ────────────────────────────────────────────────────────────
        self.model_type   = kwargs['model_type']
        self.modality     = kwargs['modality']
        
        # ── training hyper-params ──────────────────────────────────────────────────
        self.epochs       = kwargs.get("num_epochs",   30)
        self.batch_size   = kwargs.get("batch_size",   512)
        self.lr           = kwargs.get("learning_rate",1e-4)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)

        self.model_input_dim = 2*kwargs.get('tfidf_dim') 
        self.model_input_dim += kwargs.get('num_dense_cols') if kwargs.get('num_dense_cols') else 9700
        self.patience = kwargs['patience']
        layers = kwargs.get('layers', [512, 128, 64])
        layers = [self.model_input_dim] + layers
        # ── device ────────────────────────────────────────────────────────────────
        dev = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        # ── pre-processor & network ───────────────────────────────────────────────

        self._preprocessor = MultisourcePreprocessor(**kwargs)
        
        self.clf = NN(layers=layers).to(self.device)

    # --------------------------------------------------------------------- helpers
    @property
    def preprocessor(self) -> BasePreprocessor:
        return self._preprocessor

    def fit(self, X, y, X_val, y_val):
        X_tab, X_mda, X_aud = self._preprocessor.fit_transform(X)
        
        tensors = [
            torch.from_numpy(arr.astype(np.float32)).to(self.device)
            for arr in (X_tab, X_mda, X_aud)
        ]
        
        X_tensor = torch.cat(tensors, dim=1)              

        y = torch.from_numpy(np.asarray(y, np.float32)).unsqueeze(1).to(self.device)

        dataset   = TensorDataset(X_tensor, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.clf.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        self.clf.train() ; best_ap = -1
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch, target in dataloader:
                optimizer.zero_grad()

                output = self.clf(batch)
                loss   = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info("epoch %d | loss %.4f", epoch, epoch_loss / len(dataloader))

            self.evaluate(X_val, y_val, dataset="val")

            if self.ap > best_ap:
                logger.info(
                    "New best validation AP %.4f (prev %.4f) — saving model…",
                    self.ap,
                    best_ap,
                )
                best_ap = self.ap
                patience_counter = 0
                self.save()
                self.preprocessor.save()
            else:
                logger.info(
                    "Validation AP %.4f did not improve (best %.4f)", self.ap, best_ap
                )
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break
    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_proba(self, X):
        self.clf.eval()
        X_tab, X_mda, X_aud = self._preprocessor.transform(X)
        
        tensors = [
            torch.from_numpy(arr.astype(np.float32)).to(self.device)
            for arr in (X_tab, X_mda, X_aud)
        ]
        
        X_tensor = torch.cat(tensors, dim=1)  
        logits = self.clf(X_tensor)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(self, X, quantile: float = 0.99):
        probs      = self.predict_proba(X)
        threshold  = np.quantile(probs, quantile)
        return (probs > threshold).astype(int)

    # ------------------------------------------------------------------ persistence
    def save(self):
        """Save network weights AND the fitted pre-processor."""
        logger.info("Saving model to %s", self.model_path)
        torch.save(self.clf.state_dict(), self.model_path)
        logger.info("Model & pre-processor saved successfully!")

    def load(self):
        """Load network weights AND the fitted pre-processor (device-agnostic)."""
        logger.info("Loading model from %s", self.model_path)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.clf.load_state_dict(state_dict)
        self.clf.to(self.device).eval()
        logger.info("Model & pre-processor loaded successfully!")


# ────────────────────────────────────────────────────────────────────────────
class NN(nn.Module):
    """Simple feed-forward network: e.g. layers=[512,128,64] → 512→128→64→1."""
    def __init__(self, layers):
        super().__init__()
        
        net = []
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            net.append(nn.Linear(in_dim, out_dim))
            net.append(nn.ReLU(inplace=True))
        net.append(nn.Linear(layers[-1], 1))
        self.net = nn.Sequential(*net)
        print(self)

    def forward(self, x):
        return self.net(x)
