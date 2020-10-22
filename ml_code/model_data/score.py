import pandas as pd

def score_df(gs_object, X_test, y_test):
    gs_df = pd.DataFrame(data={'Best Parameters': gs_object.best_params_,
                               'Mean Score': gs_object.score(X_test, y_test),
                               'Highest Prediction Score': gs_object.best_score_
                               })
    print(gs_df)
