from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def pipeline_rfe(X_train, X_test, y_train, y_test):
    """
    #TEST_RESULTS 4/23/2020 - all unscaled
    #Selected features: ['amount', 'description', 'post_date', 'file_created_date',
    #'optimized_transaction_date', 'panel_file_created_date', 'account_score', 'amount_std_lag3']
    #--
    #TEST_RESULTS 5/6/2020 - all unscaled
    Selected features: ['description', 'post_date', 'file_created_date', 'optimized_transaction_date',
                        'panel_file_created_date', 'account_score', 'amount_std_lag3', 'amount_std_lag7']
    """

    #cols = [c for c in bank_df if bank_df[c].dtype == 'int64' or 'float64']
    #X_train = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
    #y_train = bank_df['primary_merchant_name']
    #X_test = bank_df[cols].drop(columns = ['primary_merchant_name'], axis = 1)
    #y_test = bank_df['primary_merchant_name']

    #build a logistic regression and use recursive feature elimination to exclude trivial features
    log_reg = LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1, verbose=2)
    # create the RFE model and select most striking attributes
    rfe = RFE(estimator=log_reg, n_features_to_select=8, step=1, verbose=2)
    rfe = rfe.fit(X_train, y_train)
    #selected attributes
    print('Selected features: %s' % list(X_train.columns[rfe.support_]))
    print(rfe.ranking_)
    #following df contains only significant features
    X_train_rfe = X_train[X_train.columns[rfe.support_]]
    X_test_rfe = X_test[X_test.columns[rfe.support_]]
    #log_reg_param = rfe.set_params(C = 0.01, max_iter = 200, tol = 0.001)
    return X_train_rfe, X_test_rfe


def pipeline_rfe_cv(X_train, X_test, y_train, y_test):
    """
        Application of Recursive Feature Extraction - Cross Validation
        IMPORTANT
        Accuracy: for classification problems
        Mean Squared Error(MSE); Root Mean Squared Error(RSME); R2 Score: for regression
TEST RESULTS
SGDReg
    Completeness Score
    Completeness metric of a cluster labeling given a ground truth.

        A clustering result satisfies completeness if all the data points
        that are members of a given class are elements of the same cluster.

        This metric is independent of the absolute values of the labels:
        a permutation of the class or cluster label values won't change the
        score value in any way.

        This metric is not symmetric: switching ``label_true`` with ``label_pred``
        will return the :func:`homogeneity_score` which will be different in
        general.
    Optimal number of features: 9
    Selected features: ['amount', 'description', 'post_date', 'file_created_date',
                        'optimized_transaction_date', 'panel_file_created_date',
                        'account_score', 'amount_std_lag7', 'amount_std_lag30']
    Max Error -picks all features - BUT HAS GOOD CV SCORE
    Neg Mean Squared Error - picks only one feat
    Homogeneity Score
    Optimal number of features: 9
    Selected features: ['description', 'post_date', 'file_created_date',
                        'optimized_transaction_date', 'panel_file_created_date', 'account_score',
                        'amount_mean_lag3', 'amount_std_lag3', 'amount_std_lag7']
    EVALUATION METRICS DOCUMENTATION
    https://scikit-learn.org/stable/modules/model_evaluation.html
    """

    #Use the Cross-Validation function of the RFE modul
    #accuracy describes the number of correct classifications
    #LOGISTIC REGRESSION
    est_logreg = LogisticRegression(max_iter=2000)
    #SGD REGRESSOR
    est_sgd = SGDRegressor(loss='squared_loss',
                                penalty='l1',
                                alpha=0.001,
                                l1_ratio=0.15,
                                fit_intercept=True,
                                max_iter=1000,
                                tol=0.001,
                                shuffle=True,
                                verbose=0,
                                epsilon=0.1,
                                random_state=None,
                                learning_rate='constant',
                                eta0=0.01,
                                power_t=0.25,
                                early_stopping=False,
                                validation_fraction=0.1,
                                n_iter_no_change=5,
                                warm_start=False,
                                average=False)
    #SUPPORT VECTOR REGRESSOR
    est_svr = SVR(kernel='linear',
                  C=1.0,
                  epsilon=0.01)

    #WORKS WITH LOGREG(pick r2), SGDRregressor(r2;rmse)
    rfecv = RFECV(estimator=est_logreg,
                  step=2,
                  #cross_calidation determines if clustering scorers can be used or regression based!
                  #needs to be aligned with estimator
                  cv=None,
                  scoring='completeness_score',
                  n_jobs=-1,
                  verbose=2)
    rfecv.fit(X_train, y_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    rfecv_num_features = rfecv.n_features_
    print('Selected features: %s' % list(X_train.columns[rfecv.support_]))
    rfecv_features = X_train.columns[rfecv.support_]

    #plot number of features VS. cross-validation scores
    plt.figure(figsize=(10, 7))
    plt.suptitle(f"{RFECV.get_params}")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    return rfecv_features, rfecv_num_features
