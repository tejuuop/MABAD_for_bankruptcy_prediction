"""
MABAD with XGBoost‑based leaf‑index embeddings for the tabular modality.

Key changes vs. the median‑imputation version:
    • A supervised XGBClassifier is fitted on the (pre‑processed) tabular features.
    • Each sample is represented by the indices of the leaves it falls into for every
      tree (``clf.apply``).
    • These integer indices are embedded with a small trainable lookup table
      (``LeafEmbed``).  The resulting dense vector replaces the previous MLP output.
    • Everything else (losses, training loop, etc.) remains unchanged.

Save/load now also persist the leaf‑embedding parameters and the XGBoost model.
"""

from typing import Optional
import os
import math
import joblib

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
import os
import random
import numpy as np
import torch

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

# ───────────────────────── sampler ───────────────────────────────────────────
class ProportionalPositiveSampler(Sampler):
    """Imbalanced mini‑batch sampler that fixes the positive fraction."""

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


# ───────────────────────── utilities ─────────────────────────────────────────
class MLP(nn.Module):
    """Two‑layer MLP that normalises output to unit length."""

    def __init__(self, in_dim: int, hid: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, latent),
        )

    def forward(self, x):  # (B, in_dim)
        return F.normalize(self.net(x), dim=-1)  # (B, latent)


class LeafEmbed(nn.Module):
    """Averaged embedding of per‑tree leaf indices returned by XGBoost ``apply``."""

    def __init__(self, num_trees: int, num_leaves: int, emb_dim: int):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(num_leaves + 1, emb_dim) for _ in range(num_trees)]
        )
        self.num_trees = num_trees

    def forward(self, idx):  # idx: (B, num_trees) int64
        x = 0.0
        for t, emb in enumerate(self.embs):
            x = x + emb(idx[:, t])  # (B, emb_dim)
        return x / self.num_trees  # mean pooling


# ───────────────────────── main model ────────────────────────────────────────
class MABAD(BaseModel):

    # ---------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------
    def __init__(self, **kw):
                # ─── metadata ────────────────────────────────────────────────────
        self.model_type = kw.get("model_type", "tri_sad")
        self.modality = kw.get("modality", "multisource")

        # ─── input dims / encoders ───────────────────────────────────────
        self.text_dim = kw.get("tfidf_dim", 1000)
       
        self.layers = kw.get("layers", [512, 128])
        self.latent_dim = self.layers[-1]

        # ─── hyper‑parameters ─────────────────────────────────────────
        self.margin_R2 = 0.2
        self.lambda_push = kw.get("lambda_push", 1.0)
        self.alignment_weight = kw.get("alignment_weight", 0.25)

        # ─── optimisation ────────────────────────────────────────────────
        self.epochs = kw.get("num_epochs", 30)
        self.batch_size = kw.get("batch_size", 512)
        self.lr = kw.get("learning_rate", 1e-4)
        self.weight_decay = kw.get("weight_decay", 1e-4)
        self.patience = kw.get("patience", 15)
        self.threshold = kw.get("threshold", 0.5)
       

        # ─── device & preprocessor ───────────────────────────────────────
        self.device = kw.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._pre = MultisourcePreprocessor(**kw)

        # ─── XGBoost parameters ──────────────────────────────────────────
        self.n_estimators = 100
        self.max_depth = 5
        self.xgb_learning_rate = 0.05 
       
        # Choose CPU/GPU tree method
        tree_method = "gpu_hist" if self.device.startswith("cuda") else "hist"

        self.clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.xgb_learning_rate,
            objective="binary:logistic",
            tree_method=tree_method,
            verbosity=0,
            random_state = 42
        )

        # Text encoders remain MLPs
        self.mda_enc = MLP(self.text_dim, self.layers[0], self.latent_dim).to(self.device)
        self.aud_enc = MLP(self.text_dim, self.layers[0], self.latent_dim).to(self.device)

        # Centres (frozen by default; set requires_grad=True to learn)
        self.c_tab = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False).to(
            self.device
        )
        self.c_mda = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False).to(
            self.device
        )
        self.c_aud = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False).to(
            self.device
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def preprocessor(self):
        return self._pre

    # ------------------------------------------------------------------
    # Encoding helper
    # ------------------------------------------------------------------
    def _encode(self, leaf_idx_tensor, mda_tensor, aud_tensor):
        """Return ℓ2‑normalised embeddings for each modality."""
        return (
            F.normalize(self.leaf_embed(leaf_idx_tensor), dim=-1),
            self.mda_enc(mda_tensor),
            self.aud_enc(aud_tensor),
        )


    def _alignment_masked(self, z_tab, z_mda, z_aud, mask_n):
        """Cosine alignment over normal samples only (mask_n)."""
        if mask_n.sum() == 0:
            return torch.tensor(0.0, device=z_tab.device)

        zt, zm, za = z_tab[mask_n], z_mda[mask_n], z_aud[mask_n]

        mean_cos = (
            (zt * zm).sum(1).mean()
            + (zt * za).sum(1).mean()
            + (zm * za).sum(1).mean()
        ) / 3.0
        return 1.0 - mean_cos  # alignment penalty


    def _sad_loss(self, z_tab, z_mda, z_aud, y_batch):
        d_tab = torch.norm(z_tab - self.c_tab, dim=1)
        d_mda = torch.norm(z_mda - self.c_mda, dim=1)
        d_aud = torch.norm(z_aud - self.c_aud, dim=1)

        mask_n = y_batch == 0  # normals

        loss_n = (
            (d_tab[mask_n] ** 2).mean()
            + (d_mda[mask_n] ** 2).mean()
            + (d_aud[mask_n] ** 2).mean()
        )

        mask_p = y_batch == 1  # anomalies
        
        relu = lambda d: F.relu(self.margin_R2 - d**2)

        loss_p = self.lambda_push * (
            relu(d_tab[mask_p]).mean()
            + relu(d_mda[mask_p]).mean()
            + relu(d_aud[mask_p]).mean()
        )
        
        tri = self.alignment_weight * self._alignment_masked(z_tab, z_mda, z_aud, mask_n)
        return loss_n + loss_p + tri


    def fit(self, X, y, X_val, y_val):
        
        X_tab, X_mda, X_aud = self._pre.fit_transform(X) 


        logger.info("Training XGBoost…")
        self.clf.fit(X_tab, y)
        
        leaf_idx = self.clf.apply(X_tab).astype(np.int64)  

        num_trees  = leaf_idx.shape[1]
        num_nodes  = int(leaf_idx.max()) + 1   

        self.leaf_embed = LeafEmbed(
        num_trees=num_trees,
        num_leaves=num_nodes,
        emb_dim=self.latent_dim,
            ).to(self.device)

        # ─── convert all data to tensors
        y = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        leaf_tensor = torch.from_numpy(leaf_idx).to(self.device)
        mda_tensor = torch.from_numpy(X_mda).to(self.device)
        aud_tensor = torch.from_numpy(X_aud).to(self.device)

        # ─── (re‑)initialise centres from negatives
        self.leaf_embed.train()
        self.mda_enc.train()
        self.aud_enc.train()

        with torch.no_grad():
            zt, zm, za = self._encode(
                leaf_tensor[y == 0], mda_tensor[y == 0], aud_tensor[y == 0]
            )
            self.c_tab.copy_(zt.mean(0))
            self.c_mda.copy_(zm.mean(0))
            self.c_aud.copy_(za.mean(0))
        
        with torch.no_grad():
            d_tab_mda = torch.norm(self.c_tab - self.c_mda).item()
            d_tab_aud = torch.norm(self.c_tab - self.c_aud).item()
            d_mda_aud = torch.norm(self.c_mda - self.c_aud).item()

        logger.info(f"Center distances → tab-mda: {d_tab_mda:.4f}, tab-aud: {d_tab_aud:.4f}, mda-aud: {d_mda_aud:.4f}")


        params = (
            list(self.leaf_embed.parameters())
            + list(self.mda_enc.parameters())
            + list(self.aud_enc.parameters())
            + [self.c_tab, self.c_mda, self.c_aud]
        )

        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # Sampler that balances the mini‑batches for training stability
        sampler = ProportionalPositiveSampler(
            y.detach().cpu().numpy(), batch_size=self.batch_size, pos_fraction=0.05
        )
        loader = DataLoader(torch.arange(len(y)), batch_sampler=sampler)

        best_ap = 0.0
        patience_counter = 0
        for ep in range(self.epochs):
            total_loss = 0.0
            for idx in loader:
                idx_cuda = idx.to(self.device)
                zt, zm, za = self._encode(
                    leaf_tensor[idx_cuda],
                    mda_tensor[idx_cuda],
                    aud_tensor[idx_cuda],
                )
                loss = self._sad_loss(zt, zm, za, y[idx_cuda])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            logger.info("Epoch %d | loss %.4f", ep + 1, total_loss / len(loader))
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
                logger.info("Early stopping at epoch %d", ep + 1)
                break

        self.get_decision_threshold(X_val,y_val)

    @torch.no_grad()
    def _score(self, leaf_tensor, mda_tensor, aud_tensor):
        # latent embeddings
        zt, zm, za = self._encode(leaf_tensor, mda_tensor, aud_tensor)

        # per-modality Euclidean distances from centres  →  shape (B,)
        d_tab = torch.norm(zt - self.c_tab, dim=1)
        d_mda = torch.norm(zm - self.c_mda, dim=1)
        d_aud = torch.norm(za - self.c_aud, dim=1)

        # stack → (B, 3), then take row-wise max → (B,)
        d_max = torch.stack([d_tab, d_mda, d_aud], dim=1).max(dim=1).values

        return d_max.cpu().numpy()          # NumPy array of length B


    @torch.no_grad()
    def predict_proba(self, X):
        X_tab, X_mda, X_aud = self._pre.transform(X)

        # XGBoost leaf indices
        leaf_idx = self.clf.apply(X_tab).astype(np.int64)

        leaf_tensor = torch.from_numpy(leaf_idx).to(self.device)
        mda_tensor = torch.from_numpy(X_mda).to(self.device)
        aud_tensor = torch.from_numpy(X_aud).to(self.device)

        self.leaf_embed.eval()
        self.mda_enc.eval()
        self.aud_enc.eval()

        return self._score(leaf_tensor, mda_tensor, aud_tensor)

    def predict(self, X):
        scores = self.predict_proba(X)
        return (scores > self.threshold).astype(int)


    def save(self):

        # Torch components
        torch.save(
            {
                "leaf_embed": self.leaf_embed.state_dict(),
                "mda_enc": self.mda_enc.state_dict(),
                "aud_enc": self.aud_enc.state_dict(),
                "c_tab": self.c_tab.detach(),
                "c_mda": self.c_mda.detach(),
                "c_aud": self.c_aud.detach(),
                "leaf_meta": {
                    "num_trees": len(self.leaf_embed.embs),
                    "num_leaves": self.leaf_embed.embs[0].num_embeddings - 1,
                },
            },
            os.path.join(self.model_dir_path, "tri_sad.pt"),
        )


        # XGBoost model
        joblib.dump(self.clf, os.path.join(self.model_dir_path, "xgb_clf.pkl"))

        logger.info("MABAD model saved → %s", self.model_dir_path)

    def load(self):
        state = torch.load(
            os.path.join(self.model_dir_path, "tri_sad.pt"), map_location=self.device
        )

        # ── Recreate the leaf_embed with the saved sizes
        leaf_meta = state["leaf_meta"]
        self.leaf_embed = LeafEmbed(
            num_trees=leaf_meta["num_trees"],
            num_leaves=leaf_meta["num_leaves"],
            emb_dim=self.latent_dim,
        ).to(self.device)

        # ── Load state
        self.leaf_embed.load_state_dict(state["leaf_embed"])
        self.mda_enc.load_state_dict(state["mda_enc"])
        self.aud_enc.load_state_dict(state["aud_enc"])
        self.c_tab.copy_(state["c_tab"])
        self.c_mda.copy_(state["c_mda"])
        self.c_aud.copy_(state["c_aud"])

        # XGBoost
        self.clf = joblib.load(os.path.join(self.model_dir_path, "xgb_clf.pkl"))

        logger.info("MABAD model loaded from %s", self.model_dir_path)



