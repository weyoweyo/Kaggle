# https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning
# https://www.kaggle.com/isaienkov/hyperparameters-tuning-techniques
# https://xgboost.readthedocs.io/en/stable/parameter.html

import optuna
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_model(trial):
    max_depth = trial.suggest_int("max_depth", 15, 25)
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    learning_rate = trial.suggest_uniform('learning_rate', 0.01, 1)
    gamma = trial.suggest_uniform('gamma', 0.0001, 1)
    reg_lambda = trial.suggest_uniform('reg_lambda', 0, 1)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.1, 1)
    min_child_weight=trial.suggest_uniform('min_child_weight', 0, 1)

    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        colsample_bytree=colsample_bytree,
        random_state=666,
        verbosity=2
    )
    return model