from lightgbm import LGBMClassifier

def pipeline_lgbm():

    lgb_clf = LGBMClassifier(nthread=4,
                             n_jobs=-1,
                             n_estimators=10000,
                             learning_rate=0.02,
                             num_leaves=34,
                             colsample_bytree=0.9497036,
                             subsample=0.8715623,
                             max_depth=8,
                             reg_alpha=0.041545473,
                             reg_lambda=0.0735294,
                             min_split_gain=0.0222415,
                             min_child_weight=39.3259775,
                             silent=-1
                             )

    lgb_clf.fit(X_train, y_train,
            eval_metric= 'logloss',
               verbose=200)
    y_pred = lgb_clf.predict(X_test)
    print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))

    return lgb_clf