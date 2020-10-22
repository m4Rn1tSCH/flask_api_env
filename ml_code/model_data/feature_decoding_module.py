# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:14:32 2020

@author: bill-
"""

"""
Module to apply decoded prediction to the dataframe and fill the missing values
"""
# LOCAL IMPORTS
import sys
# sys.path.append('C:/Users/bill-/OneDrive/Dokumente/Docs Bill/TA_files/functions_scripts_storage/envel-machine-learning')
import psycopg2
from psycopg2 import OperationalError
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from ml_code.model_data.spending_report_csv_function import spending_report as spending_report
from ml_code.model_data.raw_data_connection import pull_df
from ml_code.model_data.split_data_w_features import split_data_feat
from ml_code.classification_models.xgbc_class import pipeline_xgb
from ml_code.model_data.pickle_io import store_pickle, open_pickle
from ml_code.model_data.SQL_connection import insert_val_alt, create_connection, execute_read_query
import ml_code.model_data.PostgreSQL_credentials as acc


# extensions to convert numpy objects to recognizable SQL objects
def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
    
def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def adapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def adapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(np.float64, adapt_numpy_float64)
register_adapter(np.int64, adapt_numpy_int64)
register_adapter(np.float32, adapt_numpy_float32)
register_adapter(np.int32, adapt_numpy_int32)
register_adapter(np.ndarray, adapt_numpy_array)

def encoding_step(df, spending_report=False, include_lag_features=False):

    fields = ['unique_mem_id', 'optimized_transaction_date', 'amount', 'description',
              'primary_merchant_name', 'transaction_category_name', 'state',
              'city', 'transaction_base_type', 'transaction_origin']    

    try:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['optimized_transaction_date'] = pd.to_datetime(
            df['optimized_transaction_date'])
    except:
        print("date columns passed. Skipping conversion.")
        pass
    # set optimized transaction_date as index for later
    df.set_index('optimized_transaction_date', drop=False, inplace=True)

    # generate the spending report with weekly/monthly expenses and income
    if spending_report:
      create_spending_report(df=df.copy())

    try:
        # Include below if needed unique ID's later:
        # df['unique_mem_id'] = df['unique_mem_id'].astype(
        #     'str', errors='ignore')
        df['amount'] = df['amount'].astype('float64')
        df['transaction_base_type'] = df['transaction_base_type'].replace(
            to_replace=["debit", "credit"], value=[1, 0])
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    date_features = (i for i in df.columns if type(i) == 'datetime64[ns]')
    try:
        for feature in date_features:
            if df[feature].isnull().sum() == 0:
                df[feature] = df[feature].apply(lambda x: dt.timestamp(x))
            else:
                df = df.drop(columns=feature, axis=1)
                print(f"Column {feature} dropped")

    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    # dropping currency if there is only one
    # try:
    #     if len(df['currency'].value_counts()) == 1:
    #         df = df.drop(columns=['currency'], axis=1)
    # except:
    #     print("Column currency was not chosen; was dropped")
    #     pass

    target_label = 'primary_merchant_name'
    UNKNOWN_TOKEN = '<unknown>'
    label_map = {}
    unique_list = df[target_label].unique().astype('str').tolist()
    unique_list.append(UNKNOWN_TOKEN)
    le_t = LabelEncoder()
    le_t.fit(unique_list)
    label_map = dict(zip(le_t.classes_, le_t.transform(le_t.classes_)))

    encoding_features = fields
    UNKNOWN_TOKEN = '<unknown>'
    embedding_map = {}
    for feature in encoding_features:
        unique_list = df[feature].unique().astype('str').tolist()
        unique_list.append(UNKNOWN_TOKEN)
        le = LabelEncoder()
        le.fit_transform(unique_list)
        embedding_map[feature] = dict(zip(le.classes_, le.transform(le.classes_)))

        # APPLICATION TO DATASET
        df[feature] = df[feature].apply(lambda x: x if x in embedding_map[feature] else UNKNOWN_TOKEN)
        df[feature] = df[feature].map(lambda x: le.transform([x])[0] if type(x) == str else x)


    '''
    IMPORTANT
    The lagging features produce NaN for the first two rows due to unavailability
    of values
    NaNs need to be dropped to make scaling and selection of features working
    '''
    if include_lag_features:

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
    # try:
    #     df = df.drop(['unique_mem_id'], axis=1)
    # except:
    #     print("Unique Member ID could not be dropped.")
    return [df, embedding_map, label_map]



def db_filler(section, store_model=False):


    # dropped amount_mean_lag7 to avoid errors
    # keep optimized_transaction_date as it becomes index
    fields = ['unique_mem_id', 'optimized_transaction_date', 'amount', 'description',
              'primary_merchant_name', 'transaction_category_name', 'state',
              'city', 'transaction_base_type', 'transaction_origin']

    # dont put label here
    feat_list = ['description', 'transaction_category_name', 'amount', 'city',
                 'state', 'transaction_base_type', 'transaction_origin']
    # connection = create_connection(db_name=acc.YDB_name,
    #                                 db_user=acc.YDB_user,
    #                                 db_password=acc.YDB_password,
    #                                 db_host=acc.YDB_host,
    #                                 db_port=acc.YDB_port)

    # # pull data and encode
    # # section from 1 - 10
    # try:
    #     filter_query = f"(SELECT {', '.join(field for field in fields)} FROM card_record \
    #                         WHERE unique_mem_id IN \
    #                         (SELECT unique_mem_id FROM user_demographic \
    #                          ORDER BY unique_mem_id ASC limit 10000 offset {10000*(section-1)})) \
    #                         UNION ALL (SELECT {', '.join(field for field in fields)} \
    #                        FROM bank_record WHERE unique_mem_id IN \
    #                        (SELECT unique_mem_id FROM user_demographic \
    #                         ORDER BY unique_mem_id ASC limit 10000 offset {10000*(section-1)}))"
    #     transaction_query = execute_read_query(connection, filter_query)
    #     df = pd.DataFrame(transaction_query, columns=fields)
    #     print(f"{len(transaction_query)} transactions.")
    # except OperationalError as e:
    #     print(f"The error '{e}' occurred")
    #     connection.rollback

    #################################################
    # TEST DF NOT ENCODED!!
    # test_df_1 = pull_df(rng=20,
    #                   spending_report=False,
    #                   plots=False)
    # test_df_2 = pull_df(rng=22,
    #                   spending_report=False,
    #                   plots=False)
    # df = pd.concat([test_df_1, test_df_2], axis=0)
    df = pull_df(rng=22,
                 spending_report=False,
                 plots=False)
    ########################################

    for num, user in enumerate(df.groupby('unique_mem_id')):
        print(f"User: {user[0]}, {num+1}/10000 users, {round(((num+1)/10000)*100, 2)}%.")

        main_df, embedding_map, label_map = encoding_step(df=df,
                                                          spending_report=False,
                                                          include_lag_features=False)


        X_train, X_train_scaled, X_train_minmax, X_test, X_test_scaled, \
            X_test_minmax, y_train, y_test = split_data_feat(df=main_df,
                                                             features=feat_list,
                                                             test_size=0.2,
                                                             label='primary_merchant_name')

        # convert train data to ndarray to avoid feature_names mismatch error
        X_array = X_train.values
        y_array = y_train.values
        Xt_array = X_test.values
        yt_array = y_test.values
        # X_train and y_train used to train pipeline
        clf_object = pipeline_xgb(x=X_array,
                                  y=y_array,
                                  test_features=Xt_array,
                                  test_target=yt_array)

        # array object
        y_pred = clf_object.predict(Xt_array)

        merchants = []
        # label_map ('merchant' : '1')
        for i in y_pred:
            for val, enc in label_map.items():
                if enc in y_pred:
                    merchants.append(val)
                else:
                    merchants.append("unseen in training")

        if store_model:
            # store trained model as pickle
            store_pickle(model=clf_object,
                         file_name=f"trained_model_{user[0]}")


        db_name = "postgres"
        db_user = "envel"
        db_pw = "envel"
        db_host = "0.0.0.0"
        db_port = "5432"

        '''
        Always use %s placeholder for queries; psycopg2 will convert most data automatically
        '''

        try:
            connection = create_connection(db_name=db_name,
                                            db_user=db_user,
                                            db_password=db_pw,
                                            db_host=db_host,
                                            db_port=db_port)
            print("-------------")
            cursor = connection.cursor()
            sql_insert_query = """
            INSERT INTO test (test_col_2)
            VALUES (%s);
            """
            # merch_list = np.ndarray(['Tatte Bakery', 'Star Market', 'Stop n Shop', 'Auto Parts Shop',
            #               'Trader Joes', 'Insomnia Cookies'])

            # this line would take individual letters as tuples and inserts
            # values = tuple([tuple(row) for row in merchants])
            # tuple or list works
            values = merchants
            for i in values:
            # executemany() to insert multiple rows rows
            # one-element-tuple with (i, )
                cursor.execute(sql_insert_query, (i, ))

            connection.commit()
            print(len(values), "record(s) inserted successfully.")

        except (Exception, psycopg2.Error) as error:
            print("Failed inserting record; {}".format(error))

        finally:
            # closing database connection.
            if (connection):
                cursor.close()
                connection.close()
                print("Operation completed.\nPostgreSQL connection is closed.")
        print("=========================")


    return 'completed'
#%%
# df_ext, embedding_map_ext, label_map_ext = encoding_step(df=df,
#                                                   spending_report=False,
#                                                   include_lag_features=False)
db_filler(section=0,
          store_model=False)