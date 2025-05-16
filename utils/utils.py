from  models.xgboost import XGBoost
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pickle
import numpy as np


from utils.logger import  get_logger

logger = get_logger(__name__)

def split(data):
    y = data['label'] 
    X = data.drop(columns=['label'])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.666, stratify=y_temp, random_state=RANDOM_SEED
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_data(modality, subset):
    if modality == 'financial':
        if subset:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_numerical.csv'))
        else:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'numerical.csv'))
        financial.set_index('uid', inplace=True)
        return split(financial)
    elif modality == 'mda_auditor':
        if subset:
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_mda_auditor.csv'))
        else:
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'mda_auditor.csv'))
        mda_auditor.set_index('uid', inplace=True)
        return split(mda_auditor)
    elif modality == 'multimodal':
        if subset:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_numerical.csv'))
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_mda_auditor.csv'))
        else:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'numerical.csv'))
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'mda_auditor.csv'))

        financial.set_index('uid', inplace=True)
        mda_auditor.set_index('uid', inplace=True)

        financial = split(financial)
        mda_auditor = split(mda_auditor)
        
        return (financial[0], mda_auditor[0]), financial[1], (financial[2], mda_auditor[2]), financial[3], (financial[4], mda_auditor[4]), financial[5]

def get_model(**kwargs):
    model_type = kwargs['model_type']
    
    if model_type == 'XGBoost':
        return XGBoost(**kwargs)
    
