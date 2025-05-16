import click
import utils.utils as utils
from utils.logger import setup_logger, get_logger
import os
import time
from config import VAL_METRIC_TO_OPTIMIZE

@click.command()
@click.option('--model_type', type=click.Choice(['XGBoost']), default='XGBoost')
@click.option('--modality', type=click.Choice(['mda_auditor','financial', 'multimodal']), default='multimodal')
@click.option('--subset', type=bool, default = True)

#################### XGBoost #################### 
@click.option('--n_estimators', default = 100, type= int)
@click.option('--max_depth', default= 10, type= int)
@click.option('--learning_rate', default= 1.0, type= float)
@click.option('--objective', default='binary:logistic', type=click.Choice(['binary:logistic']))
@click.option('--tfidf_dim', default = 500, type=int)
@click.option('--ngram_range', default = (1,1),  type=tuple)

def exe(model_type, modality, subset, n_estimators, max_depth, learning_rate, objective, tfidf_dim, ngram_range):
    start = time.time()
    setup_logger(log_file=os.path.join("logs",f"{modality}",f"{model_type}.log"))

    logger = get_logger(__name__)
    logger.info(50*"#"+"\t"+"TRAINING AND VALIDATING"+"\t"+50*"#")
    logger.info(f"Using subset? {subset}")
    
    args = locals()

    model = utils.get_model(**args)
 
    X_train, y_train, X_val, y_val, _, _ = utils.get_data(modality, subset)
    
    model.fit(X_train, y_train)
    
    model.evaluate(X_train, y_train, dataset='train')
    model.evaluate(X_val, y_val, dataset='val')

    
    previous_best_validation_performance = model.previous_best_validation_performance()
    
    if previous_best_validation_performance < model.current_validation_performance:
        logger.info(f"New best model found based on {VAL_METRIC_TO_OPTIMIZE} metric on validation set!")
        logger.info(f"Previous was {previous_best_validation_performance}. Current is {model.current_validation_performance}")
        model.save()

    
    model.save_hyperparameters()

    end = time.time()
    elapsed = end - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    logger.info(f"Execution time: {formatted}")

if __name__ == '__main__':
    exe()
