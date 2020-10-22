'''
Yodlee dataframes encoder
FIRST STAGE: retrieve the user ID dataframe with all user IDs with given filter
  dataframe called df is being generated in the current work directory as CSV
SECOND STAGE: randomly pick a user ID; encode thoroughly and yield the df
THIRD STAGE: encode all columns to numerical values and store corresponding dictionaries
'''

from sklearn.preprocessing import LabelEncoder
from psycopg2 import OperationalError
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from collections import Counter

# FILE IMPORTS FOR FLASK
# from ml_code.model_data.SQL_connection import execute_read_query, create_connection
# import ml_code.model_data.PostgreSQL_credentials as acc
# from ml_code.model_data.spending_report_csv_function import spending_report as create_spending_report

# FILE IMPORTS FOR NOTEBOOKS
# from SQL_connection import execute_read_query, create_connection
# import PostgreSQL_credentials as acc
from spending_report_csv_function import spending_report as create_spending_report

def df_encoder(df, feature_list, spending_report=False, plots=False, include_lag_features=True):

    '''
    Parameters
    ----------
    df : pandas dataframe, pass a dataframe that is to be converted
    feature_list: list, specify feature columns for conversion.
                IMPORTANT: ONLY OPTIMIZED TRANSACTION DATE IS AUTOMATICALLY CONVERTED.
    spending_report : bool, Save a spending report in directory if True. Default is False.
    plots : bool, Plots various graphs if True. Default is False.
    include_lag_features : include lag feature 'amount' to database with 3, 7, and 30 day rolls. Default is True

    Returns
    -------
    df.
    '''

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
        #ax.title('Transaction categories of user {df[unique_mem_id][0]}')
        ax.legend(loc='center right')
        plt.show()

    try:
        # if df[col] == 'datetime64[ns]':
        #     df[col] = pd.to_datetime(df[col])
        df['optimized_transaction_date'] = pd.to_datetime(
            df['optimized_transaction_date'])
    except:
        print("columns cold not be converted.")
        pass
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

    #
    date_features = (i for i in feature_list if type(i) == 'datetime64[ns]')
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

    UNKNOWN_TOKEN = '<unknown>'
    embedding_maps = {}
    for feature in feature_list:
        unique_list = df[feature].unique().astype('str').tolist()
        unique_list.append(UNKNOWN_TOKEN)
        le = LabelEncoder()
        le.fit_transform(unique_list)
        embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

        # APPLICATION TO DATASET
        df[feature] = df[feature].apply(lambda x: x if x in embedding_maps[feature] else UNKNOWN_TOKEN)
        df[feature] = df[feature].map(lambda x: le.transform([x])[0] if type(x) == str else x)

    # dropping currency if there is only one
    try:
        if len(df['currency'].value_counts()) == 1:
            df = df.drop(columns=['currency'], axis=1)
    except:
        print("Column currency was not chosen; dropping not necessary")
        pass
    '''
    IMPORTANT
    The lagging features produce NaN for the first two rows due to unavailability
    of values
    NaNs need to be dropped to make scaling and selection of features working
    '''
    if include_lag_features:
        #FEATURE ENGINEERING
        #typical engineered features based on lagging metrics
        #mean + stdev of past 3d/7d/30d/ + rolling volume
        date_index = df.index.values
        df.reset_index(drop=True, inplace=True)
        #pick lag features to iterate through and calculate features
        lag_features = ["amount"]
        #set up time frames; how many days/months back/forth
        t1 = 3
        t2 = 7
        t3 = 30
        #rolling values for all columns ready to be processed
        df_rolled_3d = df[lag_features].rolling(window=t1, min_periods=0)
        df_rolled_7d = df[lag_features].rolling(window=t2, min_periods=0)
        df_rolled_30d = df[lag_features].rolling(window=t3, min_periods=0)

        #calculate the mean with a shifting time window
        df_mean_3d = df_rolled_3d.mean().shift(periods=1).reset_index().astype(np.float32)
        df_mean_7d = df_rolled_7d.mean().shift(periods=1).reset_index().astype(np.float32)
        df_mean_30d = df_rolled_30d.mean().shift(periods=1).reset_index().astype(np.float32)

        #calculate the std dev with a shifting time window
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

    #drop all features left with empty (NaN) values
    df = df.dropna()
    #drop user IDs to avoid overfitting with useless information
    df = df.drop(['unique_mem_id',
                  'unique_bank_account_id',
                  'unique_bank_transaction_id'], axis=1)

    return df
