import click
import utils.utils as utils
from utils.logger import setup_logger, get_logger
import os
import time
from config import VAL_METRIC_TO_OPTIMIZE


@click.command()
@click.option('--model_type', type=click.Choice(['XGBoost', 'CatBoost', 'SGD', 'DevNet','DeepSAD','CE']), default = 'CE')
@click.option('--modality', type=click.Choice(['mda_auditor', 'financial', 'multimodal']), default='financial')
@click.option('--subset', type=bool, default=True)
@click.option('--split', type=click.Choice(['stratified','time']), default='time')
################# Preprocessing #################
@click.option('--tfidf_dim', default=500, type=int)
@click.option('--imputer', default=None)
@click.option('--standardize', default=True,  type=bool)
@click.option('--ngram_range', default=(1, 1),  type=tuple)
#################### Universal ####################
@click.option('--learning_rate', default=0.03, type=float)
@click.option('--num_epochs', default=10, type=int)
@click.option('--batch_size', default=64, type=int)
@click.option('--device', type=click.Choice(['cuda', 'cpu']), default='cuda')
@click.option('--weight_decay', default=1e-6, type=float)
@click.option('--layers', callback=utils.parse_input, help="Layer sizes like '[128,64,32]' or '128,64,32'", default=[])
#################### XGBoost ####################
@click.option('--n_estimators', default=100, type=int)
@click.option('--max_depth', default=10, type=int)
@click.option('--objective', default='binary:logistic', type=click.Choice(['binary:logistic']))
#################### CatBoost ####################
@click.option('--n_iterations', default=100, type=int)
@click.option('--depth', default=10, type=int)
@click.option('--loss_function', default='Logloss', type=click.Choice(['Logloss']))
#################### SGDClassifier ####################
@click.option('--loss', default='hinge', type=click.Choice(['hinge', 'log_loss']))
@click.option('--penalty', default='l2', type=click.Choice(['l2', 'l1', 'elasticnet']))
@click.option('--max_iter', default=1000, type=int)
#################### DevNet ####################
@click.option('--l', default=5000, type=int)
@click.option('--a', default=5, type=int)
@click.option('--contamination', default=0.0001, type=float)
#################### DeepSAD ####################
@click.option('--eta', default=1.0, type=float)
@click.option('--pretrain', default=False, type=bool)
#################### CE ####################
@click.option('--feats_value_layers', callback=utils.parse_input, help="Layer sizes like '[128,64,32]' or '128,64,32'", default="[32,24]")
@click.option('--feats_context_layers', callback=utils.parse_input, help="Layer sizes like '[128,64,32]' or '128,64,32'", default="[32,24]")
@click.option('--classifier_hidden_dim', default=12, type=int)

def exe(model_type, modality, subset, n_estimators, max_depth, learning_rate, objective, tfidf_dim,
        ngram_range, imputer, standardize, n_iterations, depth, loss_function, loss, penalty, max_iter,
        num_epochs, batch_size, device, l, a, contamination, layers, weight_decay, eta, pretrain,feats_value_layers,feats_context_layers,classifier_hidden_dim,split):
    start = time.time()
    setup_logger(log_file=os.path.join(
        "logs", f"{modality}", f"{model_type}.log"))

    logger = get_logger(__name__)
    logger.info(50*"#"+"\t"+"TRAINING AND VALIDATING"+"\t"+50*"#")
    logger.info(f"Using subset? {subset}")

    args = locals()

    model = utils.init_model(**args)

    X_train, y_train, X_val, y_val, _, _ = utils.get_data(modality, subset, split)
    model.fit(X_train, y_train)

    model.evaluate(X_train, y_train, dataset='train')
    model.evaluate(X_val, y_val, dataset='val')

    previous_best_validation_performance, _ = model.previous_best_validation_performance()

    if previous_best_validation_performance < model.current_validation_performance:
        logger.info(
            f"New best model found based on {VAL_METRIC_TO_OPTIMIZE} metric on validation set!")
        logger.info(
            f"Previous was {previous_best_validation_performance}. Current is {model.current_validation_performance}")
        model.save()
        model.save_hyperparameters(qualify_for_testing=True)
    else:
        model.save_hyperparameters(qualify_for_testing=False)

    end = time.time()
    elapsed = end - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    logger.info(f"Execution time: {formatted}")
    print(f"{model.current_validation_performance}")


if __name__ == '__main__':
    exe()
