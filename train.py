import click
import utils.utils as utils
from utils.logger import setup_logger, get_logger
import os
import time
from config import VAL_METRIC_TO_OPTIMIZE, SHALLOW_MODELS, NEURAL_MODELS


@click.command()
@click.option('--model_type', type=click.Choice(SHALLOW_MODELS + NEURAL_MODELS), default='MADA' ,required=True)
@click.option('--modality', type=click.Choice(['mda_auditor', 'financial','multisource']),default ='multisource' ,required=True)
@click.option('--subset', type=bool, default=False)
@click.option('--split', type=click.Choice(['stratified','time']), default='time')
@click.option('--load_tfidfs', type=bool, default=False)
################# Preprocessing #################
@click.option('--tfidf_dim', default=1000, type=int)
@click.option('--imputer', default=None)
@click.option('--standardize', default=False,  type=bool)
@click.option('--ngram_range', default=(1, 2),  type=tuple)
@click.option('--num_dense_cols', default=None, type=int)
#################### Universal ####################
@click.option('--learning_rate', default=0.03, type=float)
@click.option('--num_epochs', default=10, type=int)
@click.option('--batch_size', default=64, type=int)
@click.option('--device', default='cuda:1')
@click.option('--weight_decay', default=1e-6, type=float)
@click.option('--layers', callback=utils.parse_input, help="Layer sizes like '[128,64,32]' or '128,64,32'", default='[512,128]')
@click.option('--patience', default=5, type=int)
#################### XGBoost ####################
@click.option('--n_estimators', default=100, type=int)
@click.option('--max_depth', default=10, type=int)
@click.option('--objective', default='binary:logistic', type=click.Choice(['binary:logistic']))
#################### CatBoost ####################
@click.option('--n_iterations', default=100, type=int)
@click.option('--depth', default=10, type=int)
@click.option('--loss_function', default='Logloss', type=click.Choice(['Logloss']))
#################### DeepSAD ####################
@click.option('--eta', default=5.0, type=float)
@click.option('--pretrain', default=False, type=bool)
#################### DevNet ####################
@click.option('--l', default=5000, type=int)
@click.option('--a', default=5, type=int)


def exe(model_type, modality, subset, n_estimators, max_depth, learning_rate, objective, tfidf_dim,
        ngram_range, imputer, standardize, n_iterations, depth, loss_function,  num_epochs, batch_size, device, layers, weight_decay
        ,split, num_dense_cols, patience, load_tfidfs, eta, pretrain, l, a):
    
    args = locals()
    start = time.time()
    setup_logger(log_file=os.path.join(
        "logs", f"{modality}", f"{model_type}.log"))

    logger = get_logger(__name__)
    logger.info(50*"#"+"\t"+"TRAINING AND VALIDATING"+"\t"+50*"#")
    logger.info(f"Using subset? {subset}")



    if load_tfidfs and modality =='multisource':
        X_train, y_train, X_val, y_val, _, _ = utils.load_fins_tfidfs() # (pd.Dataframe(),sparse,sparse)
    else:   
        X_train, y_train, X_val, y_val, _, _ = utils.get_data(modality, subset, split) # (pd.Dataframe(), pd.Dataframe(['item_7','opinion_text']))

    model = utils.init_model(**args)

    if model_type in NEURAL_MODELS:
        model.fit(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train)
        model.get_decision_threshold(X_val,y_val)
        model.evaluate(X_train, y_train, dataset='train')
        model.evaluate(X_val, y_val, dataset='val')



    if model_type in SHALLOW_MODELS:
        previous_best_validation_performance, _ = model.previous_best_validation_performance()
        if previous_best_validation_performance < model.current_validation_performance:
            logger.info(
                f"New best model found based on {VAL_METRIC_TO_OPTIMIZE} metric on validation set!")
            logger.info(
                f"Previous was {previous_best_validation_performance}. Current is {model.current_validation_performance}")
            model.save()
            model.preprocessor.save()
            model.save_hyperparameters(args, qualify_for_testing=True)
        else:
            model.save_hyperparameters(args, qualify_for_testing=False)
    else:
        model.save_hyperparameters(args, qualify_for_testing=True)

    end = time.time()
    elapsed = end - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    logger.info(f"Execution time: {formatted}")

if __name__ == '__main__':
    exe()
