from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPRegressor

def mlp_reg(X_train_minmax, X_test_minmax, y_train, y_test):
    '''
    SelectKBest and Multi-Layer Perception Regressor
    '''
    #Create pipeline with feature selector and classifier
    #learning_rate = 'adaptive'; when solver='sgd'
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2)),
        ('clf', MLPRegressor(activation='relu',
                             solver='lbfgs',
                             learning_rate='constant'))])

    #Create a parameter grid, provide the values for the models to try
    #Parameter explanation:
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    params = {
        'feature_selection__k': [4, 5, 6, 7],
        'clf__max_iter': [800, 1200, 1500],
        'clf__alpha': [0.0001, 0.001, 0.01]}

    #Initialize the grid search object
    grid_search_mlpreg = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_mlpreg.fit(X_train_minmax, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_mlpreg.best_params_}")
    print("Overall score: %.4f" %(grid_search_mlpreg.score(X_test_minmax, y_test)))
    print(f"Best accuracy with parameters: {grid_search_mlpreg.best_score_}")

    return grid_search_mlpreg



    '''
    Pipeline 8; 2020-05-20 11:50:43
    {'clf__alpha': 0.001, 'clf__max_iter': 1200, 'feature_selection__k': 5}
    Overall score: 0.9632
    Best accuracy with parameters: 0.9623355019137264
    '''
