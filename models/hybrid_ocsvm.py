import os
import math
import joblib
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM

from models.base_model import BaseModel
from utils.logger import get_logger
from preprocessing.multisource_preprocessor import MultisourcePreprocessor

logger = get_logger(__name__)

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, latent),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class LeafEmbed(nn.Module):
    def __init__(self, num_trees: int, num_leaves: int, emb_dim: int):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(num_leaves + 1, emb_dim) for _ in range(num_trees)]
        )
        self.num_trees = num_trees

    def forward(self, idx):
        x = 0.0
        for t, emb in enumerate(self.embs):
            x = x + emb(idx[:, t])
        return x / self.num_trees


class AutoEncoderSVM(BaseModel):
    def __init__(self, **kw):
        self.model_type = kw.get("model_type", "ae_ocsvm")
        self.modality = kw.get("modality", "multisource")

        self.text_dim = kw.get("tfidf_dim", 1000)
        self.layers = kw.get("layers", [512, 128])
        self.latent_dim = self.layers[-1]

        self.epochs = kw.get("num_epochs", 30)
        self.batch_size = kw.get("batch_size", 512)
        self.lr = kw.get("learning_rate", 1e-4)
        self.weight_decay = kw.get("weight_decay", 1e-4)
        self.threshold = kw.get("threshold", 0.5)
        self.device = kw.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._pre = MultisourcePreprocessor(**kw)

        self.n_estimators = 100
        self.max_depth = 5
        self.xgb_learning_rate = 0.05

        tree_method = "gpu_hist" if self.device.startswith("cuda") else "hist"
        self.clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.xgb_learning_rate,
            objective="binary:logistic",
            tree_method=tree_method,
            verbosity=0,
            random_state=42,
        )

        self.mda_enc = MLP(self.text_dim, self.layers[0], self.latent_dim).to(self.device)
        self.aud_enc = MLP(self.text_dim, self.layers[0], self.latent_dim).to(self.device)

        self.rep_dim = self.latent_dim * 3

        self.decoder = nn.Sequential(
            nn.Linear(self.rep_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.rep_dim)
        ).to(self.device)

        self.svm = None  # Will hold One-Class SVM

    @property
    def preprocessor(self):
        return self._pre

    def _encode_concat(self, leaf_tensor, mda_tensor, aud_tensor):
        z_tab = F.normalize(self.leaf_embed(leaf_tensor), dim=-1)
        z_mda = self.mda_enc(mda_tensor)
        z_aud = self.aud_enc(aud_tensor)
        h = torch.cat([z_tab, z_mda, z_aud], dim=1)
        return h

    def fit(self, X, y=None, X_val=None, y_val=None):

        pos_fraction = y.sum()/len(y)
        logger.info(f"Anomalous ratio is about {pos_fraction}")
        X_tab, X_mda, X_aud = self._pre.fit_transform(X)

        logger.info("Training XGBoost for leaf embeddings…")
        self.clf.fit(X_tab, y)
        leaf_idx = self.clf.apply(X_tab).astype(np.int64)
        num_trees, num_nodes = leaf_idx.shape[1], int(leaf_idx.max()) + 1

        self.leaf_embed = LeafEmbed(num_trees, num_nodes, self.latent_dim).to(self.device)

        leaf_tensor = torch.from_numpy(leaf_idx).to(self.device)
        mda_tensor = torch.from_numpy(X_mda).to(self.device)
        aud_tensor = torch.from_numpy(X_aud).to(self.device)

        dataset = torch.utils.data.TensorDataset(leaf_tensor, mda_tensor, aud_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        params = (
            list(self.leaf_embed.parameters()) +
            list(self.mda_enc.parameters()) +
            list(self.aud_enc.parameters()) +
            list(self.decoder.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        logger.info("Training AutoEncoder on latent space...")

        self.leaf_embed.train() ; self.mda_enc.train() ; self.aud_enc.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for leaf_b, mda_b, aud_b in loader:
                leaf_b, mda_b, aud_b = leaf_b.to(self.device), mda_b.to(self.device), aud_b.to(self.device)

                latent = self._encode_concat(leaf_b, mda_b, aud_b)
                recon = self.decoder(latent)

                loss = F.mse_loss(recon, latent)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info(f"Epoch {epoch+1} | MSE loss: {epoch_loss / len(loader):.6f}")

        # Train One-Class SVM on latent representations
        logger.info("Encoding full training set for One-Class SVM...")
        self.leaf_embed.eval() ; self.mda_enc.eval() ; self.aud_enc.eval()
        with torch.no_grad():
            latent_space = self._encode_concat(leaf_tensor, mda_tensor, aud_tensor).cpu().numpy()

        logger.info("Fitting One-Class SVM...")
        self.svm = OneClassSVM(kernel='rbf', nu=pos_fraction, gamma='scale')
        self.svm.fit(latent_space)
        logger.info("One-Class SVM training complete.")
        # self.get_decision_threshold(X_val, y_val)
        self.save()

    @torch.no_grad()
    def predict_proba(self, X):
        X_tab, X_mda, X_aud = self._pre.transform(X)
        leaf_idx = self.clf.apply(X_tab).astype(np.int64)

        leaf_tensor = torch.from_numpy(leaf_idx).to(self.device)
        mda_tensor = torch.from_numpy(X_mda).to(self.device)
        aud_tensor = torch.from_numpy(X_aud).to(self.device)

        self.leaf_embed.eval() ; self.mda_enc.eval() ; self.aud_enc.eval()
        latent = self._encode_concat(leaf_tensor, mda_tensor, aud_tensor).cpu().numpy()

        scores = -self.svm.decision_function(latent)  # Higher score = more anomalous
        return scores

    def predict(self, X):
        scores = self.predict_proba(X)
        return (scores > self.threshold).astype(int)  # or calibrate threshold empirically

    def save(self):
        # Save only PyTorch components
        torch.save({
            "leaf_embed": self.leaf_embed.state_dict(),
            "mda_enc": self.mda_enc.state_dict(),
            "aud_enc": self.aud_enc.state_dict(),
            "decoder": self.decoder.state_dict(),
            "leaf_meta": {
                "num_trees": len(self.leaf_embed.embs),
                "num_leaves": self.leaf_embed.embs[0].num_embeddings - 1,
            }
        }, os.path.join(self.model_dir_path, "ae_ocsvm.pt"))

        # Save XGBoost classifier
        joblib.dump(self.clf, os.path.join(self.model_dir_path, "xgb_clf.pkl"))

        # Save the sklearn One-Class SVM
        if self.svm is not None:
            joblib.dump(self.svm, os.path.join(self.model_dir_path, "ocsvm.pkl"))

        logger.info("AutoEncoder, XGBoost, and OCSVM models saved to %s", self.model_dir_path)


    def load(self):
        # Load PyTorch model components
        state = torch.load(os.path.join(self.model_dir_path, "ae_ocsvm.pt"), map_location=self.device)
        leaf_meta = state["leaf_meta"]

        self.leaf_embed = LeafEmbed(
            num_trees=leaf_meta["num_trees"],
            num_leaves=leaf_meta["num_leaves"],
            emb_dim=self.latent_dim
        ).to(self.device)

        self.leaf_embed.load_state_dict(state["leaf_embed"])
        self.mda_enc.load_state_dict(state["mda_enc"])
        self.aud_enc.load_state_dict(state["aud_enc"])
        self.decoder.load_state_dict(state["decoder"])

        # Load XGBoost classifier
        self.clf = joblib.load(os.path.join(self.model_dir_path, "xgb_clf.pkl"))

        # Load One-Class SVM
        svm_path = os.path.join(self.model_dir_path, "ocsvm.pkl")
        if os.path.exists(svm_path):
            self.svm = joblib.load(svm_path)
            logger.info("One-Class SVM loaded from %s", svm_path)
        else:
            self.svm = None
            logger.warning("One-Class SVM file not found — svm attribute is None.")

        logger.info("AutoEncoder + OCSVM components loaded from %s", self.model_dir_path)
