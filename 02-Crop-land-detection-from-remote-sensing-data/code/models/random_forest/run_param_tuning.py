import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import predict, metrics
from utils import dataset
from models.random_forest import param_tuning
from sklearn.model_selection import train_test_split
from datetime import datetime

# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

print("starting random forest hyperparameter tuning...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
all_x, all_y = dataset.get_train(train)

x_train, x_test, y_train, y_test = train_test_split(all_x,
                                                    all_y,
                                                    test_size=0.1,
                                                    random_state=42)
'''
# Number of trees in random forest
n_estimators = [100, 200, 300, 400, 500]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

random_search_best_params, random_search_best_score = \
    param_tuning.random_search(random_grid, x_train, y_train)
print("random_search_best_params")
print(random_search_best_params)
print("random_search_best_score")
print(random_search_best_score)
'''


'''
random_search_best_params = {'n_estimators': 300,
                             'min_samples_split': 2,
                             'min_samples_leaf': 1,
                             'max_features': 'auto',
                             'max_depth': 40,
                             'bootstrap': True}
random_search_best_score
0.8434767025089606
'''

random_search_best_params = {'n_estimators': [250, 300, 350],
                             'min_samples_split': [2],
                             'min_samples_leaf': [1],
                             'max_features': ['auto'],
                             'max_depth': [35, 40, 45],
                             'bootstrap': [True]}

'''
grid_search_best_params = param_tuning.grid_search(random_search_best_params,
                                                     x_train,
                                                     y_train,
                                                     x_test,
                                                     y_test)
print("grid_search_best_params")
print(grid_search_best_params)
'''

'''
grid_search_best_params = {'bootstrap': True, 
                           'max_depth': 45, 
                           'max_features': 'auto', 
                           'min_samples_leaf': 1, 
                           'min_samples_split': 2, 
                           'n_estimators': 350}
'''
rf_best = RandomForestClassifier(n_estimators=350,
                                 max_depth=45,
                                 max_features='auto',
                                 bootstrap=True,
                                 min_samples_leaf=1,
                                 min_samples_split=2)

y_test_pred = predict.get_prediction(x_train, y_train, x_test, rf_best, 'rf')
metrics.get_f1(y_test, y_test_pred)
metrics.get_classification_report(y_test, y_test_pred)


print("end")
end_time = datetime.now()
print(end_time)