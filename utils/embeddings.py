import pandas as pd 
import pathlib, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import os
import os
from pathlib import Path

# Go one level up from the current notebook directory
root_dir = Path.cwd().resolve().parent
os.chdir(root_dir)


from torch.nn.functional import normalize
# Go one level up from the current notebook directory
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42

def time_split(data):

    data['year'] = data['uid'].apply(lambda x : x.split('__')[1])
    data['year'] = pd.to_datetime(data['year'])
    data = data.sort_values('year').reset_index(drop=True)

    positives = data[data['label'] == 1].copy()

    n_pos = len(positives)
    train_end = int(0.65 * n_pos)
    val_end = int(0.75 * n_pos)

    train_max_date = positives.iloc[train_end]['year']
    val_max_date = positives.iloc[val_end]['year']

    train_df = data[data['year'] <= train_max_date]
    val_df = data[(data['year'] > train_max_date) & (data['year'] <= val_max_date)]
    test_df = data[data['year'] > val_max_date]


    train_df.set_index('uid', inplace=True)
    val_df.set_index('uid', inplace=True)
    test_df.set_index('uid', inplace=True)

    y_train = train_df['label'] 
    X_train = train_df.drop(columns=['label','year'])

    y_val = val_df['label'] 
    X_val = val_df.drop(columns=['label','year'])

    y_test = test_df['label'] 
    X_test = test_df.drop(columns=['label','year'])


    return X_train, y_train, X_val, y_val, X_test, y_test

def split(data):
    data.set_index('uid', inplace=True)
    y = data['label'] 
    X = data.drop(columns=['label'])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.666, stratify=y_temp, random_state=RANDOM_SEED
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_data(modality, subset, split_strategy):
    if modality == 'financial':
        if subset:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_numerical.csv'))
        else:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'numerical.csv'))
        return split(financial) if split_strategy == 'stratified' else time_split(financial)
    elif modality == 'mda_auditor':
        if subset:
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_mda_auditor.csv'))
        else:
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'mda_auditor.csv'))
        return split(mda_auditor) if split_strategy == 'stratified' else time_split(mda_auditor)
    elif modality == 'multimodal' or modality == 'multisource':
        if subset:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_numerical.csv'))
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_mda_auditor.csv'))
        else:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'numerical.csv'))
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'mda_auditor.csv'))
        financial = split(financial) if split_strategy == 'stratified' else time_split(financial)
        mda_auditor = split(mda_auditor) if split_strategy == 'stratified' else time_split(mda_auditor)
        
        return (financial[0], mda_auditor[0]), financial[1], (financial[2], mda_auditor[2]), financial[3], (financial[4], mda_auditor[4]), financial[5]



class LongformerEmbedder:
    def __init__(self, model_name="allenai/longformer-base-4096", max_len=4096, pooling="cls"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to("cuda:1")
        self.max_len = max_len
        self.pooling = pooling

    @torch.no_grad()
    def _pool(self, hidden, attention_mask):
        if self.pooling == "cls":
            return hidden[:, 0, :]                            # (B, hidden)
        else:                                                 # mean pooling
            mask = attention_mask.unsqueeze(-1)               # (B, L, 1)
            summed = (hidden * mask).sum(1)                   # (B, hidden)
            counts = mask.sum(1)                              # (B, 1)
            return summed / counts.clamp(min=1e-9)            # avoid /0
    
    @torch.no_grad()
    def encode_series_batched(
        self,
        series: pd.Series,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Encode a Series in mini-batches for speed."""
        embs = []
        for i in tqdm(range(0, len(series), batch_size), desc="Batch-encoding"):
            batch_texts = series.iloc[i : i + batch_size].astype(str).tolist()
            batch_tokens = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            ).to(self.model.device)

            hidden = self.model(**batch_tokens).last_hidden_state  # (B, L, H)
            pooled = self._pool(hidden, batch_tokens.attention_mask)      # (B, H)
            pooled = normalize(pooled, p=2, dim=1).cpu().numpy()
            embs.append(pooled)

        return np.vstack(embs).astype(np.float32)
    

    # ---------------- save / load helpers ------------------
    @staticmethod
    def save_embeddings(emb, out_path):
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_path, emb)

    @staticmethod
    def load_embeddings(path):
        return np.load(path, mmap_mode="r")


if __name__ == "__main__":
    X_train, _, X_val, _, X_test, _ = get_data(modality='mda_auditor', subset=False, split_strategy='time')

    X_train_mda, X_train_auditor = X_train.item_7, X_train.opinion_text
    X_val_mda, X_val_auditor = X_val.item_7, X_val.opinion_text
    X_test_mda, X_test_auditor = X_test.item_7, X_test.opinion_text


    # ── Create a single embedder (shared weights) ────────────────
    embedder = LongformerEmbedder(pooling="cls")   # or pooling="mean"

    # Helper to encode & save one split + modality ----------------
    def encode_and_save(series, out_path, batch_size=8):
        emb = embedder.encode_series_batched(series.astype(str), batch_size=batch_size)
        embedder.save_embeddings(emb, out_path)


    batch_size = 12
    # ── MD&A embeddings ──────────────────────────────────────────
    # encode_and_save(X_train_mda,  "data/embeddings/mda_train.npy",  batch_size=batch_size)   # MD&A is long → smaller batch
    # encode_and_save(X_val_mda,    "data/embeddings/mda_val.npy",    batch_size=batch_size)
    # encode_and_save(X_test_mda,   "data/embeddings/mda_test.npy",   batch_size=batch_size)

    # ── Auditor embeddings ───────────────────────────────────────
    encode_and_save(X_train_auditor, "data/embeddings/aud_train.npy", batch_size=batch_size)
    encode_and_save(X_val_auditor,   "data/embeddings/aud_val.npy",   batch_size=batch_size)
    encode_and_save(X_test_auditor,  "data/embeddings/aud_test.npy",  batch_size=batch_size)


