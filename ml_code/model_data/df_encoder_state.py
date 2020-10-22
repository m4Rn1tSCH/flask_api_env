# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:55:04 2020

@author: bill-
"""

from psycopg2 import OperationalError
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from SQL_connection import execute_read_query, create_connection
import PostgreSQL_credentials as acc
from spending_report_csv_function import spending_report as create_spending_report

def df_encoder_state(state, rng=4, spending_report=False, plots=False, include_lag_features=True):

    '''
    Parameters
    ----------
    state : str, Takes abbreviation of a US state.
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
    #filter_query = f"SELECT unique_mem_id, state, city, zip_code, income_class, file_created_date FROM user_demographic WHERE state = 'MA'"
    #transaction_query = execute_read_query(connection, filter_query)
    #query_df = pd.DataFrame(transaction_query,
    #                        columns=['unique_mem_id', 'state', 'city', 'zip_code', 'income_class', 'file_created_date'])
    us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

    try:
        if state in us_states:
            filter_query = f"SELECT * FROM bank_record WHERE state = '{state}'"
            transaction_query = execute_read_query(connection, filter_query)
            df = pd.DataFrame(transaction_query,
                                   columns=['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id',
                                   'amount', 'currency', 'description', 'transaction_date', 'post_date', 'transaction_base_type',
                                   'transaction_category_name', 'primary_merchant_name', 'secondary_merchant_name', 'city',
                                   'state', 'zip_code', 'transaction_origin', 'factual_category', 'factual_id', 'file_created_date',
                                   'optimized_transaction_date', 'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred',
                                   'swipe_date', 'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                                   'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
            print(f"{len(df)} transactions in {state} on record.")
            # all these columns are empty or almost empty and contain no viable information
            df = df.drop(columns=['secondary_merchant_name', 'swipe_date', 'update_type', 'is_outlier',
                                  'is_duplicate', 'change_source', 'lag', 'mcc_inferred', 'mcc_raw',
                                  'factual_id', 'factual_category', 'zip_code', 'yodlee_transaction_status',
                                  'file_created_date', 'panel_file_created_date', 'account_source_type',
                                  'account_type', 'account_score', 'user_score'], axis=1)
        else:
            print("Passed state is not valid.")
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
        ax.legend(loc='center right')
        plt.show()

    '''
    Generate a spending report of the unaltered dataframe
    Use the datetime columns just defined
    This report measures either the sum or mean of transactions happening
    '''
    # convert all date col from date to datetime objects
    # date objects will block Select K Best if not converted
    df['post_date'] = pd.to_datetime(df['post_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['optimized_transaction_date'] = pd.to_datetime(
        df['optimized_transaction_date'])

    # set optimized transaction_date as index for later
    df.set_index('optimized_transaction_date', drop=False, inplace=True)

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

    # attempt to convert date objects to unix timestamps as numeric value (fl64)
    # if they have no missing values; otherwise they are being dropped
    date_features = ['post_date', 'transaction_date', 'optimized_transaction_date']
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
    encoding_features = ['primary_merchant_name', 'city', 'state', 'description',
                         'transaction_category_name', 'transaction_origin', 'currency']
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

# pass with letter ticks!
#df = df_encoder(rng=9, state = 'VT')

