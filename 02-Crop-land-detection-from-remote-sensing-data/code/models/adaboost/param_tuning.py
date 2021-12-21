from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def random_search(random_grid, x_train, y_train):
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    ada = AdaBoostClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    ada_random = RandomizedSearchCV(estimator=ada,
                                   param_distributions=random_grid,
                                   n_iter=5,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs = -1)
    # Fit the random search model
    ada_random.fit(x_train, y_train)
    return ada_random.best_params_, ada_random.best_score_


def grid_search(param_grid, x_train, y_train, x_test, y_test):
    # Create a based model
    ada = AdaBoostClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=ada, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)
    return grid_search.best_params_

