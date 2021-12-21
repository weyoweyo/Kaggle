import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from models.xgboost.param_tuning import create_model
from utils import dataset


print("starting xgboost hyperparameter tuning...")
start_time = datetime.now()
print(start_time)


train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
all_x, all_y = dataset.get_train(train)

x_train, x_test, y_train, y_test = train_test_split(all_x,
                                                    all_y,
                                                    test_size=0.1,
                                                    random_state=42)


class Optimizer:
    def __init__(self, metric, trials=50):
        self.metric = metric
        self.trials = trials
        self.sampler = TPESampler(seed=666)

    def objective(self, trial):
        model = create_model(trial)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        if self.metric == 'acc':
            return accuracy_score(y_test, preds)
        else:
            return f1_score(y_test, preds)

    def optimize(self):
        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler
        )
        study.optimize(
            self.objective,
            n_trials=self.trials
        )
        return study.best_params


optimizer = Optimizer('f1')
optuna_params = optimizer.optimize()
optuna_params['random_state'] = 666
print(optuna_params)

print("xgboost hyperparameter tuning ends...")
end_time = datetime.now()
print(end_time)