from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def pipeline_xgb(x, y, test_features, test_target):

    '''
    x: df/ndarray. Pass training data features.
    y: df/ndarray. Pass training data label that is to be predicted.
    test_features: df/ndarray. Test data features.
    test_target: df/ndarray. Test data label that is to be predicted.
    '''

    xgbclf = XGBClassifier()
    # Add silent=True to avoid printing out updates with each cycle
    xgbclf.fit(x, y)

    # make predictions
    y_pred = xgbclf.predict(test_features)
    print("Accuracy Score: " + str(accuracy_score(y_pred, test_target)))

    return xgbclf