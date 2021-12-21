from utils import predict, metrics, dataset, submission
from datetime import datetime
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
# fine-tune parameter values reference: papers/Sentinel-2 Image Scene Classification: A Comparison between
# Sen2Cor and a Machine Learning Approach

print("starting voting classifier...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
x_train, y_train = dataset.get_train(train)
test = dataset.get_test(raw_test)

print('label percentage for training dataset: ')
print(metrics.label_percentages(y_train))


adaboost = AdaBoostClassifier(n_estimators=150)
knn = KNeighborsClassifier(n_neighbors=1)
lgbm = LGBMClassifier(max_depth=18,
                       num_leaves=143,
                       feature_fraction=0.5398760642626285,
                       bagging_fraction=0.9304436544614162,
                       learning_rate=0.06525287721325376,
                       max_bin=24,
                       min_data_in_leaf=20,
                       subsample=0.175744924178873)

rf = RandomForestClassifier(n_estimators=350,
                             max_depth=45,
                             max_features='auto',
                             bootstrap=True,
                             min_samples_leaf=1,
                             min_samples_split=2)

'''
rf = RandomForestClassifier(criterion='gini',
                            n_estimators=242,
                            max_depth=20,
                            max_features='sqrt',
                            bootstrap=True,
                            min_samples_leaf=1,
                            min_samples_split=50)
'''

'''
extratree = ExtraTreesClassifier(criterion='gini',
                            n_estimators=279,
                            max_depth=20,
                            max_features='sqrt',
                            bootstrap=True,
                            min_samples_leaf=1,
                            min_samples_split=10)
'''


xgboost = XGBClassifier(max_depth=7,
                        n_estimators=157,
                        learning_rate= 0.23989473738149508,
                        gamma= 0.7442032316202452,
                        random_state=666)

voting = VotingClassifier(estimators=[('adaboost', adaboost),
                                      ('knn', knn),
                                      ('lgbm', lgbm),
                                      ('rf', rf),
                                      ('xgboost', xgboost)
                                      ],
                          voting='soft')


y_pred = predict.get_prediction(x_train,
                                y_train,
                                test,
                                voting,
                                "soft voting")

print('label percentage for prediction: ')
n0_per, n1_per = metrics.label_percentages(y_pred)
print(n0_per)
print(n1_per)
filename = 'n0_' + str(format(n0_per, '.4f')) + '_n1_' + str(format(n1_per, '.4f'))
submission.save_data(sub, y_pred, 'predictions/', filename)

print("end")
end_time = datetime.now()
print(end_time)
