from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestRegressor

def rfr_reg(X_train, X_test, y_train, y_test, bank_df):
    '''
    SelectKBest and Random Forest Regressor
    '''
    #Create pipeline with feature selector and classifier
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('reg', RandomForestRegressor(n_estimators=75,
                                      max_depth=len(bank_df.columns)/2,
                                      min_samples_split=4))
    ])

    #Create a parameter grid, provide the values for the models to try
    params = {
        'feature_selection__k': [5, 6, 7, 8, 9],
        'reg__n_estimators': [75, 100, 150, 200],
        'reg__min_samples_split': [4, 8, 10, 15],
    }

    #Initialize the grid search object
    grid_search_rfr = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_rfr.fit(X_train, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_rfr.best_params_}")
    print("Overall score: %.4f" % (grid_search_rfr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_rfr.best_score_}")

    return grid_search_rfr



    '''
    PRIMARY_MERCHANT_NAME
    ---------
    Pipeline 3; 2020-04-29 11:13:21
    {'feature_selection__k': 7, 'reg__min_samples_split': 8, 'reg__n_estimators': 150}
    Overall score: 0.6965
    Best accuracy with parameters: 0.6820620369181245
    ---
    Pipeline 3; 2020-05-01 10:01:18
    {'feature_selection__k': 7, 'reg__min_samples_split': 4, 'reg__n_estimators': 100}
    Overall score: 0.9319
    Best accuracy with parameters: 0.9181502112642107
        CITY
    Pipeline 3; 2020-05-04 14:50:00 Full Set
    {'feature_selection__k': 7, 'reg__min_samples_split': 4, 'reg__n_estimators': 100}
    Overall score: 0.8422
    Best accuracy with parameters: 0.8558703875627366
    ---
    Pipeline 3; 2020-05-04 17:10:16 Sparse Set
    {'feature_selection__k': 7, 'reg__min_samples_split': 4, 'reg__n_estimators': 150}
    Overall score: 0.7186
    Best accuracy with parameters: 0.75653465869764
    ---
    Pipeline 3; 2020-05-06 10:13:08 with kbest features
    {'feature_selection__k': 5, 'reg__min_samples_split': 8, 'reg__n_estimators': 150}
    Overall score: 0.6255
    Best accuracy with parameters: 0.5813314519498283
    ---
    Pipeline 3; 2020-05-06 16:02:09 Amount_mean_lag7
    {'feature_selection__k': 5, 'reg__min_samples_split': 4, 'reg__n_estimators': 100}
    Overall score: 0.9641
    Best accuracy with parameters: 0.9727385020905415
    '''
