import click
import utils.utils as utils
from utils.logger import setup_logger, get_logger
import os
import time
from config import  SHALLOW_MODELS, NEURAL_MODELS

@click.command()
@click.option("--model_name", type=str, required=True)
@click.option(
    "--model_type",
    type=click.Choice(SHALLOW_MODELS + NEURAL_MODELS),
)
@click.option(
    "--modality", type=click.Choice(["mda_auditor", "financial", "multisource"])
)
@click.option("--subset", type=bool, default=False)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda")
@click.option("--split", type=click.Choice(["stratified", "time"]), default="time")
@click.option('--load_tfidfs', type=bool, default=False)
def exe(model_type, modality, subset, device, split, model_name,load_tfidfs):
    start = time.time()

    setup_logger(log_file=os.path.join("logs", f"{modality}", f"{model_type}.log"))
    logger = get_logger(__name__)

    logger.info(50 * "#" + "\t" + "TESTING" + "\t" + 50 * "#")

    args = locals()

    hyperparams_path = os.path.join(
        "models", "saved_models", modality, model_type, "hyperparameters.json"
    )
    
    model = utils.init_model_from_hyperparams(hyperparams_path, **args)


    if load_tfidfs and modality =='multisource':
        _, _, _, _, X_test, y_test = utils.load_fins_tfidfs() # (pd.Dataframe(),sparse,sparse)
    else:   
        _, _, _, _, X_test, y_test = utils.get_data(modality, subset, split) # (pd.Dataframe(), pd.Dataframe(['item_7','opinion_text']))


    model.load()
    model.preprocessor.load()
    model.evaluate(X_test, y_test, dataset="Test")
    model.save_results(model_name)

    end = time.time()
    elapsed = end - start
    formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    logger.info(f"Execution time: {formatted}")


if __name__ == "__main__":
    exe()
