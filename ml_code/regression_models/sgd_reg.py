from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDRegressor

def sgd_reg(X_train, X_test, y_train, y_test):
    '''
    SelectKBest and SGDRegressor -needs non-negative values
    '''
    #Create pipeline with feature selector and regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2)),
        ('reg', SGDRegressor(loss='squared_loss', penalty='l1'))
    ])

    #Create a parameter grid, provide the values for the models to try
    params = {
        'feature_selection__k': [5, 6, 7],
        'reg__alpha': [0.01, 0.001, 0.0001],
        'reg__max_iter': [800, 1000, 1500]
    }

    #Initialize the grid search object
    grid_search_sgd = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_sgd.fit(X_train, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_sgd.best_params_}")
    print("Overall score: %.4f" % (grid_search_sgd.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_sgd.best_score_}")

    return grid_search_sgd



    '''
    Pipeline 2; 2020-04-29 14:13:46
    {'feature_selection__k': 5, 'reg__alpha': 0.0001, 'reg__max_iter': 800}
    Overall score: -12552683945869548245665121782413383849471150345158656.0000
    Best accuracy with parameters: -1.459592722067248e+50
    '''
