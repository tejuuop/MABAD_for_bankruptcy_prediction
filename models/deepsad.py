import os
import math
import joblib
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from xgboost import XGBClassifier

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


class ProportionalPositiveSampler(Sampler):
    def __init__(self, y, batch_size: int, pos_fraction: float = 0.05):
        self.y = y
        self.batch_size = batch_size
        self.pos_fraction = pos_fraction

        self.pos_idx = np.where(self.y == 1.0)[0]
        self.neg_idx = np.where(self.y == 0.0)[0]

        self.num_pos = math.ceil(self.pos_fraction * self.batch_size)
        self.num_neg = self.batch_size - self.num_pos

    def __iter__(self):
        num_batches = len(self.neg_idx) // self.num_neg
        for _ in range(num_batches):
            pos_batch = np.random.choice(self.pos_idx, self.num_pos, replace=True)
            neg_batch = np.random.choice(self.neg_idx, self.num_neg, replace=False)
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield torch.from_numpy(batch)

    def __len__(self):
        return len(self.neg_idx) // self.num_neg


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


class DeepSAD(BaseModel):
    def __init__(self, **kw):
        self.model_type = kw.get("model_type", "deep_sad")
        self.modality = kw.get("modality", "multisource")

        self.text_dim = kw.get("tfidf_dim", 1000)
        self.layers = kw.get("layers", [512, 128])
        self.latent_dim = self.layers[-1]

        self.epochs = kw.get("num_epochs", 30)
        self.batch_size = kw.get("batch_size", 512)
        self.lr = kw.get("learning_rate", 1e-4)
        self.weight_decay = kw.get("weight_decay", 1e-4)
        self.patience = kw.get("patience", 15)
        self.threshold = kw.get("threshold", 0.5)
        self.margin = kw.get("margin", 0.5)

        self.device = kw.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._pre = MultisourcePreprocessor(**kw)

        self.n_estimators = 100
        self.max_depth = 0
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
        self.c = nn.Parameter(torch.zeros(self.rep_dim), requires_grad=False).to(self.device)

    @property
    def preprocessor(self):
        return self._pre

    def _encode(self, leaf_idx_tensor, mda_tensor, aud_tensor):
        z_tab = F.normalize(self.leaf_embed(leaf_idx_tensor), dim=-1)
        z_mda = self.mda_enc(mda_tensor)
        z_aud = self.aud_enc(aud_tensor)
        h = torch.cat([z_tab, z_mda, z_aud], dim=1)
        return h

    def _sad_loss(self, reps, y_batch, eta=1.0):
        dist = torch.norm(reps - self.c, dim=1)

        normal_loss = dist[y_batch == 0].pow(2).mean()

        anomaly_loss = (1.0 / (dist[y_batch == 1].pow(2) + 1e-6)).mean()

        return normal_loss + eta * anomaly_loss


    def fit(self, X, y, X_val=None, y_val=None):
        X_tab, X_mda, X_aud = self._pre.fit_transform(X)
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)

        logger.info("Training XGBoost…")
        self.clf.fit(X_tab, y)
        leaf_idx = self.clf.apply(X_tab).astype(np.int64)
        num_trees, num_nodes = leaf_idx.shape[1], int(leaf_idx.max()) + 1

        self.leaf_embed = LeafEmbed(num_trees, num_nodes, self.latent_dim).to(self.device)

        leaf_tensor = torch.from_numpy(leaf_idx).to(self.device)
        mda_tensor = torch.from_numpy(X_mda).to(self.device)
        aud_tensor = torch.from_numpy(X_aud).to(self.device)

        # Initialise center c with mean of normal samples
        with torch.no_grad():
            reps = self._encode(leaf_tensor, mda_tensor, aud_tensor)
            self.c.copy_(reps[y_tensor == 0].mean(0))

        params = (
            list(self.leaf_embed.parameters()) +
            list(self.mda_enc.parameters()) +
            list(self.aud_enc.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        sampler = ProportionalPositiveSampler(y_tensor.cpu().numpy(), self.batch_size, pos_fraction=0.05)
        loader = DataLoader(torch.arange(len(y_tensor)), batch_sampler=sampler)

        best_ap = -1
        for epoch in range(self.epochs):
            self.leaf_embed.train() ; self.mda_enc.train() ; self.aud_enc.train()
            total_loss = 0

            for idx in loader:
                idx_cuda = idx.to(self.device)
                reps = self._encode(
                    leaf_tensor[idx_cuda],
                    mda_tensor[idx_cuda],
                    aud_tensor[idx_cuda]
                )
                y_batch = y_tensor[idx_cuda]

                loss = self._sad_loss(reps, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info("Epoch %d | loss %.4f", epoch + 1, total_loss / len(loader))

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

        # self.get_decision_threshold(X_val,y_val)

    @torch.no_grad()
    def predict_proba(self, X):
        X_tab, X_mda, X_aud = self._pre.transform(X)
        leaf_idx = self.clf.apply(X_tab).astype(np.int64)

        leaf_tensor = torch.from_numpy(leaf_idx).to(self.device)
        mda_tensor = torch.from_numpy(X_mda).to(self.device)
        aud_tensor = torch.from_numpy(X_aud).to(self.device)

        self.leaf_embed.eval() ; self.mda_enc.eval() ; self.aud_enc.eval()
        reps = self._encode(leaf_tensor, mda_tensor, aud_tensor)
        dist = torch.norm(reps - self.c, dim=1)
        return dist.cpu().numpy()

    def predict(self, X):
        scores = self.predict_proba(X)
        return (scores > self.threshold).astype(int)

    def save(self):
        torch.save(
            {
                "leaf_embed": self.leaf_embed.state_dict(),
                "mda_enc": self.mda_enc.state_dict(),
                "aud_enc": self.aud_enc.state_dict(),
                "center": self.c.detach(),
                "leaf_meta": {
                    "num_trees": len(self.leaf_embed.embs),
                    "num_leaves": self.leaf_embed.embs[0].num_embeddings - 1,
                },
            },
            os.path.join(self.model_dir_path, "deep_sad.pt"),
        )

        joblib.dump(self.clf, os.path.join(self.model_dir_path, "xgb_clf.pkl"))
        logger.info("DeepSAD model saved → %s", self.model_dir_path)

    def load(self):
        state = torch.load(
            os.path.join(self.model_dir_path, "deep_sad.pt"), map_location=self.device
        )

        leaf_meta = state["leaf_meta"]
        self.leaf_embed = LeafEmbed(
            num_trees=leaf_meta["num_trees"],
            num_leaves=leaf_meta["num_leaves"],
            emb_dim=self.latent_dim,
        ).to(self.device)

        self.leaf_embed.load_state_dict(state["leaf_embed"])
        self.mda_enc.load_state_dict(state["mda_enc"])
        self.aud_enc.load_state_dict(state["aud_enc"])
        self.c.copy_(state["center"])

        self.clf = joblib.load(os.path.join(self.model_dir_path, "xgb_clf.pkl"))
        logger.info("DeepSAD model loaded from %s", self.model_dir_path)
