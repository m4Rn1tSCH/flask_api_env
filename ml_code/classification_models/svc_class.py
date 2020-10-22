from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

def svc_class(X_train, X_test, y_train, y_test):
    '''
    SelectKBest and Support Vector Classifier
    '''
    #Create pipeline with feature selector and classifier
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', SVC())])

    #Create a parameter grid, provide the values for the models to try
    #Parameter explanation:
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    params = {
        'feature_selection__k': [4, 5, 6, 7, 8, 9],
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__gamma': [0.1, 0.01, 0.001]}

    #Initialize the grid search object
    grid_search_svc = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_svc.fit(X_train, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_svc.best_params_}")
    print("Overall score: %.4f" % (grid_search_svc.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svc.best_score_}")

    return grid_search_svc



    '''
    Pipeline 7; 2020-04-28 10:22:10
    {'clf__C': 100, 'clf__gamma': 0.1, 'feature_selection__k': 5}
    Best accuracy with parameters: 0.6742596944770858
    ---
    Pipeline 7; 2020-04-29 10:06:28 SCALED DATA
    {'clf__C': 0.01, 'clf__gamma': 0.1, 'feature_selection__k': 4}
    Overall score: 0.3696
    Best accuracy with parameters: 0.34202115158636903
    ---
    Pipeline 7; 2020-04-29 10:11:02 UNSCALED DATA
    {'clf__C': 10, 'clf__gamma': 0.01, 'feature_selection__k': 5}
    Overall score: 0.5266
    Best accuracy with parameters: 0.5592068155111634
    ---
    Pipeline 7; 2020-04-30 11:38:13
    {'clf__C': 1, 'clf__gamma': 0.01, 'feature_selection__k': 4}
    Overall score: 0.5408
    Best accuracy with parameters: 0.5335967104732726
    ---
    Pipeline 7; 2020-05-01 10:29:08
    {'clf__C': 100, 'clf__gamma': 0.01, 'feature_selection__k': 4}
    Overall score: 0.9346
    Best accuracy with parameters: 0.9102803738317757
    ---
    Pipeline 7; 2020-05-04 10:52:47
    {'clf__C': 10, 'clf__gamma': 0.1, 'feature_selection__k': 4}
    Overall score: 0.9121
    Best accuracy with parameters: 0.9171339563862928
    ---
        CITY
    Pipeline 7; 2020-05-04 14:58:15 Full Set
    {'clf__C': 10, 'clf__gamma': 0.01, 'feature_selection__k': 5}
    Overall score: 0.8841
    Best accuracy with parameters: 0.8797507788161993
    ---
    Pipeline 7; 2020-05-04 17:14:48 Sparse Set
    {'clf__C': 10, 'clf__gamma': 0.1, 'feature_selection__k': 5}
    Overall score: 0.7533
    Best accuracy with parameters: 0.7908651132790454
    ---
        AMOUNT-MEAN-LAG7
    Pipeline 7; 2020-05-06 16:17:40
    {'clf__C': 10, 'clf__gamma': 0.001, 'feature_selection__k': 4}
    Overall score: 0.1044
    Best accuracy with parameters: 0.16726598403612028
    '''
