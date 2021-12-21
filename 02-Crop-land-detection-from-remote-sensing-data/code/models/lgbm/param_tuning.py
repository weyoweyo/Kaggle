from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# https://neptune.ai/blog/lightgbm-parameters-guide
# https://www.kaggle.com/somang1418/tuning-hyperparameters-under-10-minutes-lgbm


def create_lgbm_model(trial):
    max_depth = trial.suggest_int("max_depth", 15, 25)
    num_leaves = trial.suggest_int("num_leaves", 50, 200)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0, 1)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0, 1)
    learning_rate = trial.suggest_uniform('learning_rate', 0.001, 1)
    max_bin=trial.suggest_int("max_bin", 15, 25)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 15, 25)
    subsample=trial.suggest_uniform('subsample', 0, 1)

    model = LGBMClassifier(
        max_depth=max_depth,
        num_leaves=num_leaves,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        learning_rate=learning_rate,
        max_bin=max_bin,
        min_data_in_leaf=min_data_in_leaf,
        subsample=subsample,
        verbose=2
    )
    return model