from  models.xgboost import XGBoost
from models.catboost import CatBoost
from models.sgd import SGD
from models.devnet import DevNet
from models.deepsad import DeepSAD
from models.context_embedder import CE

import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pickle
import numpy as np
import ast

from utils.logger import  get_logger

logger = get_logger(__name__)

def time_split(data):

    data['year'] = data['uid'].apply(lambda x : x.split('__')[1])
    data['year'] = pd.to_datetime(data['year'])
    data = data.sort_values('year').reset_index()

    positives = data[data['label'] == 1].copy()

    n_pos = len(positives)
    train_end = int(0.65 * n_pos)
    val_end = int(0.75 * n_pos)

    train_max_date = positives.iloc[train_end]['year']
    val_max_date = positives.iloc[val_end]['year']

    train_df = data[data['year'] <= train_max_date]
    val_df = data[(data['year'] > train_max_date) & (data['year'] <= val_max_date)]
    test_df = data[data['year'] > val_max_date]

    logger.info(f"Train positives: {train_df['label'].sum()}")
    logger.info(f"Val positives: {val_df['label'].sum()}")
    logger.info(f"Test positives: {test_df['label'].sum()}")


    train_df.set_index('uid', inplace=True)
    val_df.set_index('uid', inplace=True)
    test_df.set_index('uid', inplace=True)

    y_train = train_df['label'] 
    X_train = train_df.drop(columns=['label','year'])

    y_val = val_df['label'] 
    X_val = val_df.drop(columns=['label','year'])

    y_test = test_df['label'] 
    X_test = test_df.drop(columns=['label','year'])

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"val shape: {X_val.shape}")
    logger.info(f"test shape: {X_test.shape}")

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
    elif modality == 'multimodal':
        if subset:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_numerical.csv'))
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'subset_mda_auditor.csv'))
        else:
            financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'numerical.csv'))
            mda_auditor = pd.read_csv(os.path.join(os.getcwd(), 'data', 'mda_auditor.csv'))
        financial = split(financial) if split_strategy == 'stratified' else time_split(financial)
        mda_auditor = split(mda_auditor) if split_strategy == 'stratified' else time_split(mda_auditor)
        
        return (financial[0], mda_auditor[0]), financial[1], (financial[2], mda_auditor[2]), financial[3], (financial[4], mda_auditor[4]), financial[5]

def init_model(**kwargs):
    model_type = kwargs['model_type']
    
    if model_type == 'XGBoost':
        return XGBoost(**kwargs)
    elif model_type == 'CatBoost':
        return CatBoost(**kwargs)
    elif model_type == 'SGD':
        return SGD(**kwargs)
    elif model_type == 'DevNet':
        return DevNet(**kwargs)
    elif model_type == 'DeepSAD':
        return DeepSAD(**kwargs)
    elif model_type == 'CE':
        return CE(**kwargs)
    
def init_model_from_hyperparams(hyperparms_path, **kwargs):
    model_type = kwargs['model_type']
    
    if model_type == 'XGBoost':
        return XGBoost.from_hyperparams(hyperparms_path)
    elif model_type == 'CatBoost':
        return CatBoost.from_hyperparams(hyperparms_path)
    elif model_type == 'SGD':
        return SGD.from_hyperparams(hyperparms_path)
    elif model_type == 'DevNet':
        return DevNet.from_hyperparams(hyperparms_path)
    elif model_type == 'DeepSAD':
        return DeepSAD.from_hyperparams(hyperparms_path)
    elif model_type == 'CE':
        return CE.from_hyperparams(hyperparms_path)
    
def parse_input(ctx, param, value):
    if value.startswith('['):
        return ast.literal_eval(value)
    else:
        return [int(x.strip()) for x in value.split(',')]