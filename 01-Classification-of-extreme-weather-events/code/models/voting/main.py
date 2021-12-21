import pandas as pd
import warnings
from utils import dataset, metric, submission

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, \
                             GradientBoostingClassifier, \
                             ExtraTreesClassifier, \
                             VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("Voting Classifier")

train, test, sub = dataset.get_from_files('train', 'test', 'sample_submission')

removed_features = ['PS']
all_features = pd.concat([train.iloc[:, :-1], test]).reset_index(drop=True)
X_train = train.iloc[:, 0:train.shape[1]-1]
y_train = train.iloc[:, -1:]
clean_data = dataset.preprocessing(all_features, removed_features)
clean_data_add_season = dataset.add_seasons(clean_data, all_features)
fea_eng_train, train_y, fea_eng_test = dataset.separate_train_test(train, clean_data_add_season)
kfold = KFold(n_splits=3, random_state=42, shuffle=True)


def accuracy_percent(model, features, label, kf):
    accuracy = cross_val_score(model, features, label, scoring='accuracy', cv=kf)
    return accuracy


models = [LogisticRegression(C=0.1, solver='newton-cg'),
          SVC(C=0.1, random_state=42),
          DecisionTreeClassifier(),
          KNeighborsClassifier(n_neighbors=19),
          RandomForestClassifier(),
          GradientBoostingClassifier(),
          ExtraTreesClassifier(),
          LGBMClassifier(),
          XGBClassifier(objective='multi:softmax', eval_metric='merror'),
         ]

scores = {}

names = ["LogisticRegression",
         "SVM",
         "DecisionTreeClassifier",
         "KNN",
         "RandomForestClassifier",
         "GradientBoostingClassifier",
         "ExtraTreesClassifier",
         "LGBM",
         "XGB"
         ]

for name, model in zip(names, models):
    score = accuracy_percent(model, fea_eng_train, train_y, kfold)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))
    scores[name] = (score.mean(), score.std())

final_models = [
          ('lr', LogisticRegression(solver='newton-cg')),           
          ('dt', DecisionTreeClassifier()),           
          ('rf', RandomForestClassifier()),           
          ('gb', GradientBoostingClassifier()),            
          ('etc', ExtraTreesClassifier()),           
          ('svc', SVC(probability=True)),           
          ('lgbm', LGBMClassifier()),           
          ('xgb', XGBClassifier())
]

eclfsoft = VotingClassifier(estimators=final_models, voting='soft')
eclfsoft = eclfsoft.fit(fea_eng_train, train_y)

labels_soft = eclfsoft.predict(fea_eng_test)

score_soft_voting = accuracy_percent(eclfsoft, fea_eng_train, train_y, kfold)

print("{}: {:.6f}, {:.4f}".format('soft voting', score_soft_voting.mean(), score_soft_voting.std()))


labels_perc = metric.label_percentages(labels_soft)
print(labels_perc)
submission.save_data(labels_soft, '../../data/predictions/', str(labels_perc))
