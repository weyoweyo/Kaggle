import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
                             GradientBoostingClassifier, \
                             ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from utils import metrics, dataset
from datetime import datetime
warnings.filterwarnings('ignore')

print("starting model comparison...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
x_train, y_train = dataset.get_train(train)

kfold = KFold(n_splits=10, random_state=42, shuffle=True)

models = [LogisticRegression(),
          RandomForestClassifier(),
          GradientBoostingClassifier(),
          ExtraTreesClassifier(),
          LGBMClassifier(),
          XGBClassifier(),
         ]

scores = {}

names = ["LogisticRegression",
         "RandomForestClassifier",
         "GradientBoostingClassifier",
         "ExtraTreesClassifier",
         "LGBM",
         "XGB"
         ]

for name, model in zip(names, models):
    score = metrics.f1_cv(model, x_train, y_train, kfold)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))
    scores[name] = (score.mean(), score.std())

print(scores)

print("end")
end_time = datetime.now()
print(end_time)
