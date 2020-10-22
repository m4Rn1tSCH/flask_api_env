from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier

def knn_class(X_train, X_test, y_train, y_test):
    '''
    SelectKBest and K Nearest Neighbor
    '''
    #Create pipeline with feature selector and classifier
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('clf', KNeighborsClassifier())])

    #Create a parameter grid, provide the values for the models to try
    params = {
        'feature_selection__k': [1, 2, 3, 4, 5, 6, 7],
        'clf__n_neighbors': [2, 3, 4, 5, 6, 7, 8]}

    #Initialize the grid search object
    grid_search_knn = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_knn.fit(X_train, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_knn.best_params_}")
    print("Overall score: %.4f" % (grid_search_knn.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_knn.best_score_}")
    
    return grid_search_knn



    '''
    Pipeline 6; 2020-04-27 11:00:27
    {'clf__n_neighbors': 7, 'feature_selection__k': 3}
    Best accuracy with parameters: 0.5928202115158637
    ------
    Pipeline 6; 2020-04-29 10:01:21 WITH SCALED DATA
    {'clf__n_neighbors': 4, 'feature_selection__k': 3}
    Overall score: 0.3696
    Best accuracy with parameters: 0.6156286721504113
    -------
    Pipeline 6; 2020-05-01 10:21:01
    {'clf__n_neighbors': 2, 'feature_selection__k': 4}
    Overall score: 0.9243
    Best accuracy with parameters: 0.9015576323987539
    -------
        CITY
    Pipeline 6; 2020-05-04 14:51:44 Full Set
    {'clf__n_neighbors': 2, 'feature_selection__k': 3}
    Overall score: 0.9028
    Best accuracy with parameters: 0.9071651090342681
    ---
    Pipeline 6; 2020-05-04 17:12:14 Sparse Set
    {'clf__n_neighbors': 3, 'feature_selection__k': 5}
    Overall score: 0.6926
    Best accuracy with parameters: 0.7287349834717407
    ---
        AMOUNT_MEAN_LAG7
    Pipeline 6; 2020-05-06 16:15:12
    {'clf__n_neighbors': 2, 'feature_selection__k': 1}
    Overall score: 0.1157
    Best accuracy with parameters: 0.154583568491494
    '''