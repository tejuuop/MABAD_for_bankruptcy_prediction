RANDOM_SEED = 42
VAL_METRIC_TO_OPTIMIZE = ["auc", "ap","fbeta","recall_at_precision"][1]
SHALLOW_MODELS = ['XGBoost', 'CatBoost',]
NEURAL_MODELS = ['MLP', 'MABAD','DeepSAD', 'DevNet','OCSVM']