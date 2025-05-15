import click
import utils.utils as utils
from utils.logger import setup_logger, get_logger
import os
import time


@click.command()
@click.option('--model_type', type=click.Choice(['XGBoost']), default='XGBoost')
@click.option('--modality', type=click.Choice(['mda_auditor','financial', 'multimodal']), default='financial')
@click.option('--subset', type=bool, default = True)

#################### XGBoost #################### 
@click.option('--n_estimators', default = 100, type= int)
@click.option('--max_depth', default= 10, type= int)
@click.option('--learning_rate', default= 1.0, type= float)
@click.option('--objective', default='binary:logistic', type=click.Choice(['binary:logistic']))

def exe(model_type, modality, subset, n_estimators, max_depth, learning_rate, objective):
    start = time.time()
    setup_logger(log_file=os.path.join("logs",f"{modality}_{model_type}.log"))

    logger = get_logger(__name__)
    logger.info(50*"#"+"\t"+"TRAINING AND VALIDATING"+"\t"+50*"#")
    logger.info(f"Using subset? {subset}")
    args = locals()
    model = utils.get_model(**args)
    X_train, y_train, X_val, y_val, _, _ = utils.get_data(modality, subset)
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
    logger.info(f"Positives - Train: {y_train.sum()}, Validation: {y_val.sum()}")
    model.fit(X_train, y_train)
    model.evaluate(X_val, y_val)
    
    results = {
            "model": model.model_type,
            "auc": model.auc,
            "ap": model.ap,
            "fbeta": model.fbeta,
            "recall_at_precision": model.recall_at_precision
    }

    logger.info(f"Validation set results: {results}")
    model.save_hyperparameters()
    end = time.time()
    elapsed = end - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    logger.info(f"Execution time: {formatted}")

if __name__ == '__main__':
    exe()
