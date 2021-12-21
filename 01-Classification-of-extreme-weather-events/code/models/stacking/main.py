import pandas as pd
import warnings

from sklearn.pipeline import make_pipeline

from utils import dataset, metric, submission
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
                             GradientBoostingClassifier, \
                             ExtraTreesClassifier, \
                             VotingClassifier, \
                             AdaBoostClassifier, \
                             BaggingClassifier, \
                             StackingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

print("Stacking Classifier")

train, test, sub = dataset.get_from_files('train', 'test', 'sample_submission')

removed_features = ['PS', 'PRECT', 'T200']

all_features = pd.concat([train.iloc[:, :-1], test]).reset_index(drop=True)
#X_train = train.iloc[:, 0:train.shape[1]-1]
#y_train = train.iloc[:, -1:]
#clean_data = dataset.preprocessing(all_features, removed_features)
#clean_data_add_season = dataset.add_seasons(clean_data, all_features)
#fea_eng_train, train_y, fea_eng_test = dataset.separate_train_test(train, clean_data_add_season)

clean_data = dataset.preprocessing_without_scaling(all_features, removed_features)
clean_data_add_season = dataset.add_seasons(clean_data, all_features)
fea_eng_train, train_y, fea_eng_test = dataset.separate_train_test(train, clean_data_add_season)
kfold = KFold(n_splits=3, random_state=42, shuffle=True)
print(fea_eng_train.columns)
'''
# preprocessing training and test dataset separately
all_features = pd.concat([train.iloc[:, :-1], test]).reset_index(drop=True)
X_train = train.iloc[:, 0:train.shape[1]-1]
y_train = train.iloc[:, -1:]
clean_data_training = dataset.preprocessing(X_train, removed_features)
fea_eng_train = dataset.add_seasons(clean_data_training, X_train)
clean_data_test = dataset.preprocessing(test, removed_features)
fea_eng_test = dataset.add_seasons(clean_data_test, test)
train_y = y_train
'''

models = [
          ('xgb',  XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='merror',
                    colsample_bytree=1,
                    learning_rate=0.02,
                    max_depth=4,
                    n_estimators=10
                    )
          ),
          ('gb',  GradientBoostingClassifier()),
          ('svc', SVC(C=0.1, random_state=42)),
          ('rf', RandomForestClassifier(max_depth=7))
          ]


new_models = [
          ('xgb',  XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='merror',
                    colsample_bytree=1,
                    learning_rate=0.02,
                    max_depth=4,
                    n_estimators=10
                    )
          ),
          ('gb',  GradientBoostingClassifier()),
          ('svc', make_pipeline(StandardScaler(), SVC(C=0.1, random_state=42))),
          ('rf', RandomForestClassifier(max_depth=7))
          ]


clr = StackingClassifier(
    estimators=new_models, final_estimator=LogisticRegression(C=0.1, multi_class='multinomial', solver='lbfgs', verbose=1)
)

print("Data done...")

clr.fit(fea_eng_train, train_y)

print("Training done...")

labels_stacking = clr.predict(fea_eng_test)

print("Labels...")

labels_perc = metric.label_percentages(labels_stacking)
print(labels_perc)

submission.save_data(labels_stacking, '../../data/predictions/', str(labels_perc))




