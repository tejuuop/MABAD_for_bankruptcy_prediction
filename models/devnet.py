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


class DevNet(BaseModel):
    def __init__(self, **kw):
        self.model_type = kw.get("model_type", "devnet")
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
        self.a = kw.get("devnet_a", 5.0)

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
        self.head = nn.Linear(self.latent_dim * 3, 1).to(self.device)

    @property
    def preprocessor(self):
        return self._pre

    def _encode(self, leaf_idx_tensor, mda_tensor, aud_tensor):
        return (
            F.normalize(self.leaf_embed(leaf_idx_tensor), dim=-1),
            self.mda_enc(mda_tensor),
            self.aud_enc(aud_tensor),
        )

    def _forward_score(self, leaf_tensor, mda_tensor, aud_tensor):
        z_tab, z_mda, z_aud = self._encode(leaf_tensor, mda_tensor, aud_tensor)
        h = torch.cat([z_tab, z_mda, z_aud], dim=1)
        return self.head(h).squeeze(1)

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

        positives = (y_tensor == 1).nonzero(as_tuple=True)[0]
        negatives = (y_tensor == 0).nonzero(as_tuple=True)[0]

        params = (
            list(self.leaf_embed.parameters()) +
            list(self.mda_enc.parameters()) +
            list(self.aud_enc.parameters()) +
            list(self.head.parameters())
        )
        optimizer = torch.optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay)

        best_ap = -1
        for epoch in range(self.epochs):
            self.leaf_embed.train() ; self.mda_enc.train() ; self.aud_enc.train()
            epoch_loss = 0

            neg_indices = negatives[torch.randperm(len(negatives))]
            for i in range(0, len(neg_indices), self.batch_size):
                neg_batch_idx = neg_indices[i:i+self.batch_size]
                pos_batch_idx = positives[torch.randint(0, len(positives), (len(neg_batch_idx),))]

                neg_scores = self._forward_score(
                    leaf_tensor[neg_batch_idx], mda_tensor[neg_batch_idx], aud_tensor[neg_batch_idx]
                )
                pos_scores = self._forward_score(
                    leaf_tensor[pos_batch_idx], mda_tensor[pos_batch_idx], aud_tensor[pos_batch_idx]
                )

                all_scores = torch.cat([neg_scores, pos_scores])
                mean, std = all_scores.mean(), all_scores.std()

                dev_pos = (pos_scores - mean) / std
                dev_neg = (neg_scores - mean) / std

                pos_loss = torch.where(self.a - dev_pos > 0, self.a - dev_pos, torch.zeros_like(dev_pos))
                neg_loss = torch.abs(dev_neg)

                loss = torch.cat([pos_loss, neg_loss]).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            logger.info("Epoch %d | loss %.4f", epoch + 1, epoch_loss)
            
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

        self.get_decision_threshold(X_val,y_val)

    @torch.no_grad()
    def predict_proba(self, X):
        X_tab, X_mda, X_aud = self._pre.transform(X)
        leaf_idx = self.clf.apply(X_tab).astype(np.int64)

        leaf_tensor = torch.from_numpy(leaf_idx).to(self.device)
        mda_tensor = torch.from_numpy(X_mda).to(self.device)
        aud_tensor = torch.from_numpy(X_aud).to(self.device)

        self.leaf_embed.eval() ; self.mda_enc.eval() ; self.aud_enc.eval()
        scores = self._forward_score(leaf_tensor, mda_tensor, aud_tensor)
        return scores.cpu().numpy()

    def predict(self, X):
        scores = self.predict_proba(X)
        return (scores > self.threshold).astype(int)

    def save(self):
        torch.save(
            {
                "leaf_embed": self.leaf_embed.state_dict(),
                "mda_enc": self.mda_enc.state_dict(),
                "aud_enc": self.aud_enc.state_dict(),
                "head": self.head.state_dict(),
                "leaf_meta": {
                    "num_trees": len(self.leaf_embed.embs),
                    "num_leaves": self.leaf_embed.embs[0].num_embeddings - 1,
                },
            },
            os.path.join(self.model_dir_path, "devnet_model.pt"),
        )

        joblib.dump(self.clf, os.path.join(self.model_dir_path, "xgb_clf.pkl"))
        logger.info("DevNet model saved → %s", self.model_dir_path)

    def load(self):
        state = torch.load(
            os.path.join(self.model_dir_path, "devnet_model.pt"), map_location=self.device
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
        self.head.load_state_dict(state["head"])

        self.clf = joblib.load(os.path.join(self.model_dir_path, "xgb_clf.pkl"))
        logger.info("DevNet model loaded from %s", self.model_dir_path)
