from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

def tlr_reg(X_train, X_test, y_train, y_test):
    '''
    Transformed Linear Regression
    #n_quantiles needs to be smaller than the number of samples (standard is 1000)
    '''
    transformer = QuantileTransformer(
        n_quantiles=750, output_distribution='normal')
    regressor = LinearRegression(n_jobs=-1)

    #Initialize the transformed target regressor
    regr = TransformedTargetRegressor(regressor=regressor,
                                      transformer=transformer)
    regr.fit(X_train, y_train)

    # raw LinearRegressor for comparison
    raw_target_regr = LinearRegression(n_jobs=-1).fit(X_train, y_train)

    #Print the best value combination
    print('q-t R2-score: {0:.3f}'.format(regr.score(X_test, y_test)))
    print('unprocessed R2-score: {0:.3f}'.format(raw_target_regr.score(X_test, y_test)))

    return regr, raw_target_regr



    '''
    AMOUNT_MEAN_LAG7
    q-t R2-score: 0.896
    unprocessed R2-score: 0.926
    '''
