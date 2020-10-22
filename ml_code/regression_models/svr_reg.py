from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVR

def svr_reg(X_train_minmax, X_test_minmax, y_train, y_test):
    '''
    Logistic Regression and Support Vector Kernel -needs non-negative values
    '''
    #Create pipeline with feature selector and regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2)),
        ('reg', SVR(kernel='linear'))
    ])

    #Create a parameter grid, provide the values for the models to try
    params = {
        'feature_selection__k': [4, 6, 7, 8, 9],
        'reg__C': [1.0, 0.1, 0.01, 0.001],
        'reg__epsilon': [0.30, 0.25, 0.15, 0.10],
    }

    #Initialize the grid search object
    grid_search_svr = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_svr.fit(X_train_minmax, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_svr.best_params_}")
    print("Overall score: %.4f" %(grid_search_svr.score(X_test_minmax, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svr.best_score_}")

    return grid_search_svr



    '''
    ---------
    Pipeline 4; 2020-05-01 10:06:03
    {'feature_selection__k': 8, 'reg__C': 0.1, 'reg__epsilon': 0.3}
    Overall score: 0.1292
    Best accuracy with parameters: 0.08389477382390549
    --------
        AMOUNT_MEAN_LAG7
    Pipeline 4; 2020-05-06 16:13:22
    {'feature_selection__k': 4, 'reg__C': 1.0, 'reg__epsilon': 0.1}
    Overall score: 0.6325
    Best accuracy with parameters: 0.5934902153570164
    '''
