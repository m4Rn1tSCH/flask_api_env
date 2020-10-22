from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression

def lr_class(X_train, X_test, y_train, y_test):
    '''
    SelectKBest and Logistic Regression (non-neg only)
    '''
    #Create pipeline with feature selector and regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=chi2)),
        ('reg', LogisticRegression(random_state=15))])

    #Create a parameter grid, provide the values for the models to try
    params = {
        'feature_selection__k': [5, 6, 7, 8, 9],
        'reg__max_iter': [800, 1000],
        'reg__C': [10, 1, 0.1]
    }

    #Initialize the grid search object
    grid_search_lr = GridSearchCV(pipe, param_grid=params, n_jobs=-1, verbose=2)
    grid_search_lr.fit(X_train, y_train)

    #Print the best value combination
    print(f"Best paramaters: {grid_search_lr.best_params_}")
    print("Overall score: %.4f" % (grid_search_lr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_lr.best_score_}")

    return grid_search_lr



    '''
    PRIMARY_MERCHANT_NAME
    Pipeline 1; 2020-04-29 11:02:06
    {'feature_selection__k': 5, 'reg__max_iter': 800}
    Overall score: 0.3696
    Best accuracy with parameters: 0.34202115158636903
    Pipeline 1; 2020-05-01 09:44:29
    {'feature_selection__k': 8, 'reg__max_iter': 800}
    Overall score: 0.5972
    Best accuracy with parameters: 0.605607476635514
        CITY
    Pipeline 1; 2020-05-04 14:38:23 Full Set
    {'feature_selection__k': 8, 'reg__max_iter': 800}
    Overall score: 0.7953
    Best accuracy with parameters: 0.8155763239875389
    ----
    Pipeline 1; 2020-05-04 17:00:59 Sparse Set
    {'feature_selection__k': 5, 'reg__max_iter': 800}
    Overall score: 0.4706
    Best accuracy with parameters: 0.5158026283963557

    #SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
    #F_CLASSIFIER;FOR CLASSIFICATION TASKS determines features based on the f-values between features & labels;
    #Chi2: for regression tasks; requires non-neg values
    #other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression

    takes unscaled numerical so far and minmax scaled arguments
    #numerical and minmax scaled leads to the same results being picked
    f_classif for classification tasks
    chi2 for regression tasks
    '''
