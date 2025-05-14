from  baselines.xgboost import XGBoost
import pandas as pd 
import os 

def get_data(modality):
    if modality == 'financial':
        financial = pd.read_csv(os.path.join(os.getcwd(), 'data', 'sec_fins_sparse.csv'))
        financial.set_index('uid', inplace=True)
        return financial

def get_model(**kwargs):
    model = kwargs['model_type']
    
    if model == 'XGBoost':
        return XGBoost(**kwargs)
    
