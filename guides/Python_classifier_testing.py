#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:29:02 2020

@author: bill
"""

from psycopg2 import OperationalError
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest , chi2, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from Python_SQL_connection import execute_read_query, create_connection
import PostgreSQL_credentials as acc
from Python_spending_report_csv_function import spending_report as create_spending_report
from Python_df_encoder_state import df_encoder_state

def df_encoder(rng=4, spending_report=False, plots=False, include_lag_features=True):

    '''
    Parameters
    ----------
    rng : int, Random Seed for user picker. The default is 4.
    spending_report : bool, Save a spending report in directory if True. Default is False.
    plots : bool, Plots various graphs if True. Default is False.
    include_lag_features : include lag feature 'amount' to database with 3, 7, and 30 day rolls. Default is True
    Returns
    -------
    df.
    '''

    connection = create_connection(db_name=acc.YDB_name,
                                   db_user=acc.YDB_user,
                                   db_password=acc.YDB_password,
                                   db_host=acc.YDB_host,
                                   db_port=acc.YDB_port)

    # establish connection to get user IDs for all users in MA
    filter_query = f"SELECT unique_mem_id, state, city, zip_code, income_class, file_created_date FROM user_demographic WHERE state = 'MA'"
    transaction_query = execute_read_query(connection, filter_query)
    query_df = pd.DataFrame(transaction_query,
                            columns=['unique_mem_id', 'state', 'city', 'zip_code', 'income_class', 'file_created_date'])

    try:
        for i in pd.Series(query_df['unique_mem_id'].unique()).sample(n=1, random_state=rng):
            filter_query = f"SELECT * FROM bank_record WHERE unique_mem_id = '{i}'"
            transaction_query = execute_read_query(connection, filter_query)
            df = pd.DataFrame(transaction_query,
                                   columns=['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id',
                                   'amount', 'currency', 'description', 'transaction_date', 'post_date', 'transaction_base_type',
                                   'transaction_category_name', 'primary_merchant_name', 'secondary_merchant_name', 'city',
                                   'state', 'zip_code', 'transaction_origin', 'factual_category', 'factual_id', 'file_created_date',
                                   'optimized_transaction_date', 'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred',
                                   'swipe_date', 'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                                   'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
            print(f"User {i} has {len(df)} transactions on record.")
            # all these columns are empty or almost empty and contain no viable information
            df = df.drop(columns=['secondary_merchant_name', 'swipe_date', 'update_type', 'is_outlier', 'is_duplicate',
                                            'change_source', 'lag', 'mcc_inferred', 'mcc_raw', 'factual_id', 'factual_category',
                                            'zip_code', 'yodlee_transaction_status'], axis=1)
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback

    '''
    Plotting of various relations
    The Counter object keeps track of permutations in a dictionary which can then be read and
    used as labels
    '''
    if plots:
        # Pie chart States
        state_ct = Counter(list(df['state']))
        # The * operator can be used in conjunction with zip() to unzip the list.
        labels, values = zip(*state_ct.items())
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax = plt.subplots(figsize=(20, 12))
        ax.pie(values, labels=labels, autopct='%1.1f%%',
              shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        #ax.title('Transaction locations of user {df[unique_mem_id][0]}')
        ax.legend(loc='center right')
        plt.show()

        # Pie chart transaction type
        trans_ct = Counter(list(df['transaction_category_name']))
        # The * operator can be used in conjunction with zip() to unzip the list.
        labels_2, values_2 = zip(*trans_ct.items())
        #Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax = plt.subplots(figsize=(20, 12))
        ax.pie(values_2, labels=labels_2, autopct='%1.1f%%',
              shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        ax.title('Transaction categories')
        ax.legend(loc='center right')
        plt.show()

    '''
    Generate a spending report of the unaltered dataframe
    Use the datetime columns just defined
    This report measures either the sum or mean of transactions happening
    on various days of the week/or wihtin a week or a month  over the course of the year
    '''
    # convert all date col from date to datetime objects
    # date objects will block Select K Best if not converted
    # first conversion from date to datetime objects; then conversion to unix
    df['post_date'] = pd.to_datetime(df['post_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['optimized_transaction_date'] = pd.to_datetime(
        df['optimized_transaction_date'])
    df['file_created_date'] = pd.to_datetime(df['file_created_date'])
    df['panel_file_created_date'] = pd.to_datetime(
        df['panel_file_created_date'])

    # set optimized transaction_date as index for later
    df.set_index('optimized_transaction_date', drop=False, inplace=True)

    # generate the spending report with the above randomly picked user ID
    if spending_report:
      create_spending_report(df=df.copy())

    '''
    After successfully loading the data, columns that are of no importance have been removed and missing values replaced
    Then the dataframe is ready to be encoded to get rid of all non-numerical data
    '''
    try:
        # Include below if need unique ID's later:
        # df['unique_mem_id'] = df['unique_mem_id'].astype(
        #     'str', errors='ignore')
        # df['unique_bank_account_id'] = df['unique_bank_account_id'].astype(
        #     'str', errors='ignore')
        # df['unique_bank_transaction_id'] = df['unique_bank_transaction_id'].astype(
        #     'str', errors='ignore')
        df['amount'] = df['amount'].astype('float64')
        df['transaction_base_type'] = df['transaction_base_type'].replace(
            to_replace=["debit", "credit"], value=[1, 0])
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    # attempt to convert date objects to unix timestamps as numeric value (fl64) if they have no missing values; otherwise they are being dropped
    date_features = ['post_date', 'transaction_date',
                     'optimized_transaction_date', 'file_created_date', 'panel_file_created_date']
    try:
        for feature in date_features:
            if df[feature].isnull().sum() == 0:
                df[feature] = df[feature].apply(lambda x: dt.timestamp(x))
            else:
                df = df.drop(columns=feature, axis=1)
                print(f"Column {feature} dropped")

    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    '''
    The columns PRIMARY_MERCHANT_NAME; CITY, STATE, DESCRIPTION, TRANSACTION_CATEGORY_NAME, CURRENCY
    are encoded manually and cleared of empty values
    '''
    encoding_features = ['primary_merchant_name', 'city', 'state', 'description', 'transaction_category_name', 'transaction_origin', 'currency']
    UNKNOWN_TOKEN = '<unknown>'
    embedding_maps = {}
    for feature in encoding_features:
        unique_list = df[feature].unique().astype('str').tolist()
        unique_list.append(UNKNOWN_TOKEN)
        le = LabelEncoder()
        le.fit_transform(unique_list)
        embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

        # APPLICATION TO OUR DATASET
        df[feature] = df[feature].apply(lambda x: x if x in embedding_maps[feature] else UNKNOWN_TOKEN)
        df[feature] = df[feature].map(lambda x: le.transform([x])[0] if type(x) == str else x)

    # dropping currency if there is only one
    if len(df['currency'].value_counts()) == 1:
        df = df.drop(columns=['currency'], axis=1)

    '''
    IMPORTANT
    The lagging features produce NaN for the first two rows due to unavailability
    of values
    NaNs need to be dropped to make scaling and selection of features working
    '''
    if include_lag_features:
        # FEATURE ENGINEERING
        # typical engineered features based on lagging metrics
        # mean + stdev of past 3d/7d/30d/ + rolling volume
        date_index = df.index.values
        df.reset_index(drop=True, inplace=True)
        #pick lag features to iterate through and calculate features
        lag_features = ["amount"]
        #set up time frames; how many days/months back/forth
        t1 = 3
        t2 = 7
        t3 = 30
        # rolling values for all columns ready to be processed
        df_rolled_3d = df[lag_features].rolling(window=t1, min_periods=0)
        df_rolled_7d = df[lag_features].rolling(window=t2, min_periods=0)
        df_rolled_30d = df[lag_features].rolling(window=t3, min_periods=0)

        # calculate the mean with a shifting time window
        df_mean_3d = df_rolled_3d.mean().shift(periods=1).reset_index().astype(np.float32)
        df_mean_7d = df_rolled_7d.mean().shift(periods=1).reset_index().astype(np.float32)
        df_mean_30d = df_rolled_30d.mean().shift(periods=1).reset_index().astype(np.float32)

        # calculate the std dev with a shifting time window
        df_std_3d = df_rolled_3d.std().shift(periods=1).reset_index().astype(np.float32)
        df_std_7d = df_rolled_7d.std().shift(periods=1).reset_index().astype(np.float32)
        df_std_30d = df_rolled_30d.std().shift(periods=1).reset_index().astype(np.float32)

        for feature in lag_features:
            df[f"{feature}_mean_lag{t1}"] = df_mean_3d[feature]
            df[f"{feature}_mean_lag{t2}"] = df_mean_7d[feature]
            df[f"{feature}_mean_lag{t3}"] = df_mean_30d[feature]
            df[f"{feature}_std_lag{t1}"] = df_std_3d[feature]
            df[f"{feature}_std_lag{t2}"] = df_std_7d[feature]
            df[f"{feature}_std_lag{t3}"] = df_std_30d[feature]

        df.set_index(date_index, drop=False, inplace=True)

    # drop all features left with empty (NaN) values
    df = df.dropna()
    # drop user IDs to avoid overfitting with useless information
    df = df.drop(['unique_mem_id',
                  'unique_bank_account_id',
                  'unique_bank_transaction_id'], axis=1)

    if plots:
        # seaborn plots
        ax_desc = df['description'].astype('int64', errors='ignore')
        ax_amount = df['amount'].astype('int64',errors='ignore')
        sns.pairplot(df)
        sns.boxplot(x=ax_desc, y=ax_amount)
        sns.heatmap(df)

    return df

def split_data(df, features, test_size=0.2, label='city'):


    '''
    Parameters
    ----------
    df : dataframe to split into label, features and train, test sets
    features : str, list; pick the features of the data frame.
    test_size : num from 0 - 1, the size of test set relative to train set. Default is 0.2
    label : column on dataframe to use as label. Default is 'city'
    Returns
    -------
    [X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax, y_train, y_test]
    '''
    # drop target variable in feature df
    #model_features = df.drop(labels=label, axis=1)
    model_features = df[features]
    model_label = df[label]

    if label == 'amount_mean_lag7':
        # To round of amount and lessen data complexity
        if model_label.dtype == 'float32':
            model_label = model_label.astype('int32')
        elif model_label.dtype == 'float64':
            model_label = model_label.astype('int64')
        else:
            print("model label has unsuitable data type!")

    # splitting data into train and test values
    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        random_state=7,
                                                        shuffle=True,
                                                        test_size=test_size)

    print(f"Shapes X_train:{X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    #STD SCALING
    #standard scaler works only with maximum 2 dimensions
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    min_max_scaler = MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.transform(X_test)

    return [X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax, y_train, y_test]

def pipeline_logreg():

    '''
    f_classif for classification tasks
    chi2 for regression tasks
    '''


    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('reg', LogisticRegression(random_state = 15))])

    params = {
        'feature_selection__k':[5, 6, 7, 8, 9],
        'reg__max_iter':[800, 1000],
        'reg__C':[10, 1, 0.1]
        }


    grid_search_lr = GridSearchCV(pipe, param_grid = params)



    print(f"Pipeline logreg; {dt.today()}")
    print(grid_search_lr.fit(X_train_minmax, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_lr.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_lr.best_score_}")

    return grid_search_lr

def pipeline_knn():

    '''
    Pipeline - SelectKBest and K Nearest Neighbor
    '''
    # Create pipeline with feature selector and classifier
    # replace with gradient boosted at this point or regressor
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', KNeighborsClassifier())])

    params = {
        'feature_selection__k':[1, 2, 3, 4, 5, 6, 7, 8],
        'clf__n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9],
        'clf__weights':['uniform', 'distance']}


    grid_search_knn = GridSearchCV(pipe, param_grid = params)


    print(f"Pipeline knn; {dt.today()}")
    print(grid_search_knn.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_knn.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_knn.best_score_}")
    return grid_search_knn

def pipeline_svc():

    '''
    SelectKBest and Support Vector Classifier
    '''
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', SVC())])
    # Parameter explanation
    #    C: penalty parameter
    #    gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    params = {
        'feature_selection__k':[4, 5, 6, 7, 8, 9],
        'clf__C':[0.01, 0.1, 1, 10],
        'clf__gamma':[0.1, 0.01, 0.001]}


    grid_search_svc = GridSearchCV(pipe, param_grid = params)


    print(f"Pipeline svc; {dt.today()}")
    print(grid_search_svc.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_svc.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_svc.best_score_}")
    return grid_search_svc

def pipeline_mlp():

    '''
    Pipeline 8 - SelectKBest and Multi-Layer Perceptron
    # # # # # # # # # #
    Pipeline 7; 2020-05-06 10:20:51 CITY (sgd + adaptive learning rate)
    {'clf__alpha': 0.0001, 'clf__max_iter': 2000, 'feature_selection__k': 5}
    Overall score: 0.2808
    Best accuracy with parameters: 0.26102555833266144
    '''

    # learning_rate = 'adaptive'; when solver='sgd'
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = chi2)),
        ('clf', MLPClassifier(activation='relu',
                              solver='lbfgs',
                              learning_rate='constant'))])

    # Create a parameter grid
    # Parameter explanation
    #    C: penalty parameter
    #    gamma: [standard 'auto' = 1/n_feat], kernel coefficient
    params = {
        'feature_selection__k':[4, 5, 6, 7],
        'clf__max_iter':[1500, 2000],
        'clf__alpha':[0.0001, 0.001]}


    grid_search_mlp = GridSearchCV(pipe, param_grid = params)


    print(f"Pipeline 7; {dt.today()}")
    print(grid_search_mlp.fit(X_train_minmax, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_mlp.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_mlp.best_score_}")
    return grid_search_mlp

def pipeline_rfc():

    '''
    Pipeline - SelectKBest and Random Forest Classifier
    '''

    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func = f_classif)),
        ('clf', RandomForestClassifier(n_jobs = -1,
                                      verbose=2,
                                      min_samples_split = 4))
        ])

    params = {
        'feature_selection__k':[5, 6, 7, 8, 9],
        'clf__n_estimators':[75, 100, 150, 200],
        'clf__max_depth':[4, 8, 10, 15],
        }

    grid_search_rfc = GridSearchCV(pipe, param_grid = params)

    print(f"Pipeline rfc; {dt.today()}")
    print(grid_search_rfc.fit(X_train, y_train).best_params_)
    print("Overall score: %.4f" %(grid_search_rfc.score(X_test, y_test)))
    print(f"Best accuracy with parameters: {grid_search_rfc.best_score_}")

    return grid_search_rfc

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

def pipeline_xgb():


    xgbclf = XGBClassifier(verbose=0)
    # Add silent=True to avoid printing out updates with each cycle
    xgbclf.fit(X_train, y_train, verbose=False)

    # make predictions
    y_pred = xgbclf.predict(X_test)
    print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))

    return xgbclf

# feature lists
svc_feat_merch = ['description', 'transaction_category_name', 'city',
                  'transaction_origin']
# dropped: panel_file_created_date
rfc_feat_merch = ['description', 'transaction_category_name', 'city', 'amount',
                  'state', 'transaction_origin']
# dropped: panel_file_created_date, user_score, account_score
xgb_feat_merch = ['description', 'transaction_category_name', 'amount', 'state',
                  'city', 'transaction_base_type', 'transaction_origin',
                  'amount_mean_lag7']
knn_feat_city = ['description', 'primary_merchant_name', 'amount',
                 'transaction_category_name', 'state']
svc_feat_city = ['description', 'primary_merchant_name', 'amount',
                 'transaction_category_name', 'state']
# dropped: panel_file_created_date
lgbm_feat_city = ['state', 'description', 'transaction_origin', 'amount',
                  'primary_merchant_name', 'transaction_category_name',
                  'transaction_date', 'amount_mean_lag30', 'amount_std_lag7',
                  'amount_mean_lag3', 'amount_std_lag30', 'amount_std_lag3']

#%%
# no improvement
# df = df_encoder_state(state = 'VT', rng=9, include_lag_features=True)
df = df_encoder(rng=9, include_lag_features=False)
X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax,\
y_train, y_test = split_data(df=df,
                             features=svc_feat_merch,
                             test_size=0.2,
                             label='primary_merchant_name')
pipeline_svc()

#%%
# OLD: 90.3%; NEW: 90.4%
df = df_encoder_state(state = 'VT', rng=9, include_lag_features=True)
# df_nolag = df_encoder(rng=9, include_lag_features=False)
X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax,\
y_train, y_test = split_data(df=df,
                             features=rfc_feat_merch,
                             test_size=0.2,
                             label='primary_merchant_name')
pipeline_rfc()

#%%
# attempted; takes ages (2h+
# OLD: 6.54; NEW: 5.23
df = df_encoder_state(state = 'VT', rng=9, include_lag_features=True)
# df_nolag = df_encoder(rng=9, include_lag_features=False)
X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax,\
y_train, y_test = split_data(df=df,
                             features=xgb_feat_merch,
                             test_size=0.2,
                             label='primary_merchant_name')
pipeline_xgb()

#%%
df = df_encoder_state(state = 'VT', rng=9, include_lag_features=True)
# df_nolag = df_encoder(rng=9, include_lag_features=False)
X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax,\
y_train, y_test = split_data(df=df,
                             features=knn_feat_city,
                             test_size=0.2,
                             label='city')
pipeline_knn()

#%%
# OLD: 84.9%; NEW: 87.9%
df = df_encoder_state(state = 'ME', rng=9, include_lag_features=True)
# df_nolag = df_encoder(rng=9, include_lag_features=False)
X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax,\
y_train, y_test = split_data(df=df,
                             features=svc_feat_city,
                             test_size=0.2,
                             label='city')
pipeline_rfc()

#%%
# OLD: 2.54; NEW: 7.7
df = df_encoder_state(state = 'VT', rng=9, include_lag_features=True)
# df_nolag = df_encoder(rng=9, include_lag_features=False)
X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, X_test_minmax,\
y_train, y_test = split_data(df=df,
                             features=lgbm_feat_city,
                             test_size=0.2,
                             label='city')
pipeline_lgbm()