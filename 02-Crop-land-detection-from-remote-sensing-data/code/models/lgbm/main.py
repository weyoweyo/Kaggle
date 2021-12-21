from lightgbm import LGBMClassifier
from utils import predict, metrics
from utils import dataset
from sklearn.model_selection import train_test_split
from datetime import datetime


print("starting lgbm training...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
all_x, all_y = dataset.get_train(train)

x_train, x_test, y_train, y_test = train_test_split(all_x,
                                                    all_y,
                                                    test_size=0.1,
                                                    random_state=42)

'''
{'max_depth': 18, 'num_leaves': 143, 'feature_fraction': 0.5398760642626285, 
'bagging_fraction': 0.9304436544614162, 'learning_rate': 0.06525287721325376, 
'max_bin': 24, 'min_data_in_leaf': 20, 'subsample': 0.175744924178873}'''


lgbm_best = LGBMClassifier(max_depth=18,
                           num_leaves=143,
                           feature_fraction=0.5398760642626285,
                           bagging_fraction=0.9304436544614162,
                           learning_rate=0.06525287721325376,
                           max_bin=24,
                           min_data_in_leaf=20,
                           subsample=0.175744924178873)

y_test_pred = predict.get_prediction(x_train, y_train, x_test, lgbm_best, 'lgbm')
metrics.get_f1(y_test, y_test_pred)
metrics.get_classification_report(y_test, y_test_pred)