from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier

def mlp_class(X_train, X_test, y_train, y_test):
    '''
    SelectKBest and Multi-Layer Perceptron
    '''
    #Create pipeline with feature selector and classifier
    #learning_rate = 'adaptive'; when solver='sgd'
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2)),
        ('clf', MLPClassifier(activation='relu',
                              solver='lbfgs',
                              learning_rate='constant'))])

    #Create a parameter grid, provide the values for the models to try
    #Parameter explanation:
    #   C: penalty parameter
    #   gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    params = {
        'feature_selection__k': [4, 5, 6, 7],
        'clf__max_iter': [1500, 2000],
        'clf__alpha': [0.0001, 0.001]}

    #Initialize the grid search object
    grid_search_mlp = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_mlp.fit(X_train, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_mlp.best_params_}")
    print("Overall score: %.4f" % (grid_search_mlp.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_mlp.best_score_}")

    return grid_search_mlp



    '''
    Pipeline 7; 2020-05-06 10:20:51 CITY (sgd + adaptive learning rate)
    {'clf__alpha': 0.0001, 'clf__max_iter': 2000, 'feature_selection__k': 5}
    Overall score: 0.2808
    Best accuracy with parameters: 0.26102555833266144
    '''
