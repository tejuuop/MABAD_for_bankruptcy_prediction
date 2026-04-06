import pandas as pd
import numpy as np
import xgboost as xgb
import os
import shutil
import gc
from sklearn.metrics import average_precision_score, roc_auc_score
from models.mabad import MABAD
from utils.utils import get_data
import warnings
warnings.filterwarnings('ignore')

def get_anomaly_scores(model_kwargs, weight_dir, X_train, X_test):
    """Initializes model, hot-swaps weights, and extracts scores."""
    print(f"Loading model with {model_kwargs['tfidf_dim']} dimensions...")
    
    # HOT-SWAP
    target_dir = os.path.join('models', 'saved_models', 'multisource', 'MABAD')
    os.makedirs(target_dir, exist_ok=True)
    
    shutil.copy(os.path.join(weight_dir, 'tri_sad.pt'), os.path.join(target_dir, 'tri_sad.pt'))
    if os.path.exists(os.path.join(weight_dir, 'xgb_clf.pkl')):
        shutil.copy(os.path.join(weight_dir, 'xgb_clf.pkl'), os.path.join(target_dir, 'xgb_clf.pkl'))
    
    model = MABAD(**model_kwargs)
    model.load()
    
    print("Calibrating internal data rules (Memory-Optimized)...")
    model.preprocessor.fit_transform(X_train)
    
    print("Extracting anomaly scores for training set...")
    train_scores = model.predict_proba(X_train)
    print("Extracting anomaly scores for test set...")
    test_scores = model.predict_proba(X_test)
    
    return train_scores, test_scores

def run_ensemble():
    print("=== Phase 1: Loading Raw Data ===")
    X_train, y_train, X_val, y_val, X_test, y_test = get_data('multisource', False, 'time')

    print("\n=== Memory Rescue: Freeing RAM ===")
    # 1. Delete Validation Set to free ~1GB of RAM instantly
    del X_val, y_val
    gc.collect()

    # 2. Compress the massive financial matrices from 64-bit to 32-bit (Frees ~1.5GB)
    print("Compressing financial data to float32...")
    float64_cols_train = X_train[0].select_dtypes(include=['float64']).columns
    X_train[0][float64_cols_train] = X_train[0][float64_cols_train].astype(np.float32)
    
    float64_cols_test = X_test[0].select_dtypes(include=['float64']).columns
    X_test[0][float64_cols_test] = X_test[0][float64_cols_test].astype(np.float32)

    print("\n=== Phase 2: Harvesting Level 1 Model Scores ===")
    
    # 1. Get TF-IDF Scores
    tfidf_kwargs = {
        'model_type': 'MABAD', 'modality': 'multisource', 
        'tfidf_dim': 1000, 'device': 'cuda:0', 'load_tfidfs': False, 'standardize': False,
        'ngram_range': (1, 2), 'imputer': 'mean'
    }
    tfidf_train_scores, tfidf_test_scores = get_anomaly_scores(
        tfidf_kwargs, 'models/saved_models/tfidf_weights/', X_train, X_test
    )

    # 2. Get FinBERT Scores
    finbert_kwargs = {
        'model_type': 'MABAD', 'modality': 'multisource', 
        'tfidf_dim': 768, 'device': 'cuda:0', 'load_tfidfs': False, 'standardize': False,
        'ngram_range': (1, 2), 'imputer': 'mean'
    }
    finbert_train_scores, finbert_test_scores = get_anomaly_scores(
        finbert_kwargs, 'models/saved_models/finbert_weights/', X_train, X_test
    )

    print("\n=== Phase 3: Building the Meta-Dataset ===")
    # Safely extract ONLY the numeric numbers for the final judge, ignoring text
    X_train_num = X_train[0].select_dtypes(include=[np.number]).values
    X_test_num = X_test[0].select_dtypes(include=[np.number]).values
    
    # Stack the original financials with the two new AI scores
    X_train_meta = np.column_stack((X_train_num, tfidf_train_scores, finbert_train_scores))
    X_test_meta = np.column_stack((X_test_num, tfidf_test_scores, finbert_test_scores))

    print("\n=== Phase 4: Training the Final XGBoost Judge ===")
    meta_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]) 
    )
    meta_model.fit(X_train_meta, y_train)

    print("\n=== Phase 5: Final Evaluation ===")
    final_predictions = meta_model.predict_proba(X_test_meta)[:, 1]
    
    final_ap = average_precision_score(y_test, final_predictions)
    final_auc = roc_auc_score(y_test, final_predictions)
    
    print("="*40)
    print(" 🚀 META-ENSEMBLE RESULTS 🚀")
    print("="*40)
    print(f"Final AUC: {final_auc:.4f}")
    print(f"Final Average Precision (AP): {final_ap:.4f}")
    print("="*40)

    print("\n=== Phase 6: Saving the Evidence ===")
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({'Model': ['MABAD_Meta_Ensemble'], 'AUC': [final_auc], 'AP': [final_ap]})
    results_df.to_csv('results/ensemble_final_scores.csv', index=False)
    print("✅ Scores securely saved to results/ensemble_final_scores.csv")

if __name__ == '__main__':
    run_ensemble()