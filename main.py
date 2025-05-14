import click
import utils


@click.command()
@click.option('--model_type', default='XGBoost', type=click.Choice(['XGBoost']))
@click.option('--modality', default='financial', type=click.Choice(['mda_auditor','financial', 'multimodal']))


#################### XGBoost #################### 
@click.option('--n_estimators', default = 100, type= int)
@click.option('--max_depth', default= 10, type= int)
@click.option('--learning_rate', default= 1.0, type= float)
@click.option('--objective', default='binary:logistic', type=click.Choice(['binary:logistic']))

def exe(model_type, modality,  n_estimators, max_depth, learning_rate, objective):

    model = utils.get_model(**locals())
    data = utils.get_data(modality)
    model.run(data)

if __name__ == '__main__':
    exe()
