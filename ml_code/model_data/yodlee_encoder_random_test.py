'''
Yodlee dataframes encoder
FIRST STAGE: retrieve the user ID dataframe with all user IDs with given filter
  dataframe called bank_df is being generated in the current work directory as CSV
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
import seaborn as sns

# FILE IMPORTS FOR FLASK
from ml_code.model_data.SQL_connection import execute_read_query, create_connection
import ml_code.model_data.PostgreSQL_credentials as acc
from ml_code.model_data.spending_report_csv_function import spending_report as create_spending_report

# FILE IMPORTS FOR NOTEBOOKS
# from SQL_connection import execute_read_query, create_connection
# import PostgreSQL_credentials as acc
# from spending_report_csv_function import spending_report as create_spending_report

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
    bank_df.
    '''

    connection = create_connection(db_name=acc.YDB_name,
                                   db_user=acc.YDB_user,
                                   db_password=acc.YDB_password,
                                   db_host=acc.YDB_host,
                                   db_port=acc.YDB_port)

    # establish connection to get user IDs for all users in MA
    filter_query = f"SELECT unique_mem_id, state, city, income_class FROM user_demographic WHERE state = 'MA'"
    transaction_query = execute_read_query(connection, filter_query)
    query_df = pd.DataFrame(transaction_query,
                            columns=['unique_mem_id', 'state', 'city', 'income_class'])

    # dateframe to gather bank data from one randomly chosen user
    # test user 1= 4
    # test user 2= 8
    try:
        for i in pd.Series(query_df['unique_mem_id'].unique()).sample(n=1, random_state=rng):
            print(i)
            filter_query = f"SELECT * FROM bank_record WHERE unique_mem_id = '{i}'"
            transaction_query = execute_read_query(connection, filter_query)
            bank_df = pd.DataFrame(transaction_query,
                                   columns=['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id',
                                   'amount', 'currency', 'description', 'transaction_date', 'post_date', 'transaction_base_type',
                                   'transaction_category_name', 'primary_merchant_name', 'secondary_merchant_name', 'city',
                                   'state', 'zip_code', 'transaction_origin', 'factual_category', 'factual_id', 'file_created_date',
                                   'optimized_transaction_date', 'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred',
                                   'swipe_date', 'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                                   'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
            print(f"User {i} has {len(bank_df)} transactions on record.")
            #all these columns are empty or almost empty and contain no viable information
            bank_df = bank_df.drop(columns=['secondary_merchant_name', 'swipe_date',
                                            'update_type', 'is_outlier', 'is_duplicate',
                                            'change_source', 'lag', 'mcc_inferred',
                                            'mcc_raw', 'factual_id', 'factual_category',
                                            'zip_code', 'yodlee_transaction_status',
                                            'file_created_date', 'panel_file_created_date',
                                            'account_type', 'account_source_type',
                                            'account_score'], axis=1)
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
        state_ct = Counter(list(bank_df['state']))
        # The * operator can be used in conjunction with zip() to unzip the list.
        labels, values = zip(*state_ct.items())
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax = plt.subplots(figsize=(20, 12))
        ax.pie(values, labels=labels, autopct='%1.1f%%',
              shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        #ax.title('Transaction locations of user {bank_df[unique_mem_id][0]}')
        ax.legend(loc='center right')
        plt.show()

        # Pie chart transaction type
        trans_ct = Counter(list(bank_df['transaction_category_name']))
        # The * operator can be used in conjunction with zip() to unzip the list.
        labels_2, values_2 = zip(*trans_ct.items())
        #Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax = plt.subplots(figsize=(20, 12))
        ax.pie(values_2, labels=labels_2, autopct='%1.1f%%',
              shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.axis('equal')
        #ax.title('Transaction categories of user {bank_df[unique_mem_id][0]}')
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
    bank_df['post_date'] = pd.to_datetime(bank_df['post_date'])
    bank_df['transaction_date'] = pd.to_datetime(bank_df['transaction_date'])
    bank_df['optimized_transaction_date'] = pd.to_datetime(
        bank_df['optimized_transaction_date'])
    bank_df['file_created_date'] = pd.to_datetime(bank_df['file_created_date'])
    bank_df['panel_file_created_date'] = pd.to_datetime(
        bank_df['panel_file_created_date'])

    # set optimized transaction_date as index for later
    bank_df.set_index('optimized_transaction_date', drop=False, inplace=True)

    # generate the spending report with the above randomly picked user ID
    if spending_report:
      create_spending_report(df=bank_df.copy())

    '''
    After successfully loading the data, columns that are of no importance have been removed and missing values replaced
    Then the dataframe is ready to be encoded to get rid of all non-numerical data
    '''
    try:
        # Include below if need unique ID's later:
        # bank_df['unique_mem_id'] = bank_df['unique_mem_id'].astype(
        #     'str', errors='ignore')
        # bank_df['unique_bank_account_id'] = bank_df['unique_bank_account_id'].astype(
        #     'str', errors='ignore')
        # bank_df['unique_bank_transaction_id'] = bank_df['unique_bank_transaction_id'].astype(
        #     'str', errors='ignore')
        bank_df['amount'] = bank_df['amount'].astype('float64')
        bank_df['transaction_base_type'] = bank_df['transaction_base_type'].replace(
            to_replace=["debit", "credit"], value=[1, 0])
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    # attempt to convert date objects to unix timestamps as numeric value (fl64) if they have no missing values; otherwise they are being dropped
    date_features = ['post_date', 'transaction_date',
                     'optimized_transaction_date', 'file_created_date', 'panel_file_created_date']
    try:
        for feature in date_features:
            if bank_df[feature].isnull().sum() == 0:
                bank_df[feature] = bank_df[feature].apply(lambda x: dt.timestamp(x))
            else:
                bank_df = bank_df.drop(columns=feature, axis=1)
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
        unique_list = bank_df[feature].unique().astype('str').tolist()
        unique_list.append(UNKNOWN_TOKEN)
        le = LabelEncoder()
        le.fit_transform(unique_list)
        embedding_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

        # APPLICATION TO OUR DATASET
        bank_df[feature] = bank_df[feature].apply(lambda x: x if x in embedding_maps[feature] else UNKNOWN_TOKEN)
        bank_df[feature] = bank_df[feature].map(lambda x: le.transform([x])[0] if type(x) == str else x)

    # dropping currency if there is only one
    if len(bank_df['currency'].value_counts()) == 1:
        bank_df = bank_df.drop(columns=['currency'], axis=1)

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
        date_index = bank_df.index.values
        bank_df.reset_index(drop=True, inplace=True)
        #pick lag features to iterate through and calculate features
        lag_features = ["amount"]
        #set up time frames; how many days/months back/forth
        t1 = 3
        t2 = 7
        t3 = 30
        #rolling values for all columns ready to be processed
        bank_df_rolled_3d = bank_df[lag_features].rolling(window=t1, min_periods=0)
        bank_df_rolled_7d = bank_df[lag_features].rolling(window=t2, min_periods=0)
        bank_df_rolled_30d = bank_df[lag_features].rolling(window=t3, min_periods=0)

        #calculate the mean with a shifting time window
        bank_df_mean_3d = bank_df_rolled_3d.mean().shift(periods=1).reset_index().astype(np.float32)
        bank_df_mean_7d = bank_df_rolled_7d.mean().shift(periods=1).reset_index().astype(np.float32)
        bank_df_mean_30d = bank_df_rolled_30d.mean().shift(periods=1).reset_index().astype(np.float32)

        #calculate the std dev with a shifting time window
        bank_df_std_3d = bank_df_rolled_3d.std().shift(periods=1).reset_index().astype(np.float32)
        bank_df_std_7d = bank_df_rolled_7d.std().shift(periods=1).reset_index().astype(np.float32)
        bank_df_std_30d = bank_df_rolled_30d.std().shift(periods=1).reset_index().astype(np.float32)

        for feature in lag_features:
            bank_df[f"{feature}_mean_lag{t1}"] = bank_df_mean_3d[feature]
            bank_df[f"{feature}_mean_lag{t2}"] = bank_df_mean_7d[feature]
            bank_df[f"{feature}_mean_lag{t3}"] = bank_df_mean_30d[feature]
            bank_df[f"{feature}_std_lag{t1}"] = bank_df_std_3d[feature]
            bank_df[f"{feature}_std_lag{t2}"] = bank_df_std_7d[feature]
            bank_df[f"{feature}_std_lag{t3}"] = bank_df_std_30d[feature]

        bank_df.set_index(date_index, drop=False, inplace=True)

    #drop all features left with empty (NaN) values
    bank_df = bank_df.dropna()
    #drop user IDs to avoid overfitting with useless information
    bank_df = bank_df.drop(['unique_mem_id',
                            'unique_bank_account_id',
                            'unique_bank_transaction_id'], axis=1)

    if plots:
        # seaborn plots
        ax_desc = bank_df['description'].astype('int64', errors='ignore')
        ax_amount = bank_df['amount'].astype('int64',errors='ignore')
        sns.pairplot(bank_df)
        sns.boxplot(x=ax_desc, y=ax_amount)
        sns.heatmap(bank_df)

    return bank_df
