import click
import utils.utils as utils
from utils.logger import setup_logger, get_logger
import os
import time


@click.command()
@click.option("--model_name", type=str, required=True)
@click.option(
    "--model_type",
    type=click.Choice(["XGBoost", "CatBoost", "SGD", "DevNet", "DeepSAD", "TabPFN"]),
)
@click.option(
    "--modality", type=click.Choice(["mda_auditor", "financial", "multimodal"])
)
@click.option("--subset", type=bool, default=False)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda")
@click.option("--split", type=click.Choice(["stratified", "time"]), default="time")
@click.option("--num_dense_cols", default=None)
def exe(model_type, modality, subset, device, split, model_name, num_dense_cols):
    start = time.time()

    setup_logger(log_file=os.path.join("logs", f"{modality}", f"{model_type}.log"))
    logger = get_logger(__name__)

    logger.info(50 * "#" + "\t" + "TESTING" + "\t" + 50 * "#")

    args = locals()

    hyperparams_path = os.path.join(
        "models", "saved_models", modality, model_type, "hyperparameters.json"
    )
    model = utils.init_model_from_hyperparams(hyperparams_path, **args)

    _, _, _, _, X_test, y_test = utils.get_data(modality, subset, split)

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
