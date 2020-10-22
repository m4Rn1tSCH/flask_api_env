#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:29:58 2020

@author: bill
"""

from sklearn.preprocessing import LabelEncoder
import psycopg2
from psycopg2 import OperationalError
import pandas as pd
from datetime import datetime as dt

# FILE IMPORTS FOR NOTEBOOKS
from ml_code.model_data.SQL_connection import execute_read_query, create_connection
import ml_code.model_data.PostgreSQL_credentials as acc
from ml_code.classification_models.svc_class import svc_class
from ml_code.classification_models.knn_class import knn_class
# FILE IMPORTS FOR NOTEBOOKS
from SQL_connection import execute_read_query, create_connection
import PostgreSQL_credentials as acc



def df_encoder(df, sel_feature='primary_merchant_name'):
    '''
    Returns
    -------
    [df, df_null, embedding_maps[sel_feature]]
    '''

    # first conversion from date to datetime objects; then conversion to unix
    df['panel_file_created_date'] = pd.to_datetime(
        df['panel_file_created_date'])
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
        df['unique_mem_id'] = df['unique_mem_id'].astype(
            'str', errors='ignore')
        df['unique_bank_account_id'] = df['unique_bank_account_id'].astype(
            'str', errors='ignore')
        df['unique_bank_transaction_id'] = df['unique_bank_transaction_id'].astype(
            'str', errors='ignore')
        df['amount'] = df['amount'].astype('float64')
        df['transaction_base_type'] = df['transaction_base_type'].replace(
            to_replace=["debit", "credit"], value=[1, 0])
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    # attempt to convert date objects to unix timestamps as numeric value (fl64) if they have no missing values; otherwise they are being dropped
    date_features = ['optimized_transaction_date', 'panel_file_created_date']
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

    # dropping currency if there is only one
    if len(df['currency'].value_counts()) == 1:
        df = df.drop(columns=['currency'], axis=1)
        encoding_features.remove('currency')

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

    #drop all features left with empty (NaN) values
    df = df.dropna()

    # extract rows with empty value of selected feature into own db
    df_null = df[df[sel_feature] == 0]
    df = df[df[sel_feature] != 0]

    return [df, df_null, embedding_maps[sel_feature]]


from sklearn.model_selection import train_test_split

# df without any missing values
# df_null with missing values in feat or label column



def split_data(full_df, null_df, label='primary_merchant_name'):
    model_features = full_df.drop(labels=label, axis=1)
    model_label = full_df[label]

    X_train, X_test, y_train, y_test = train_test_split(model_features,
                                                        model_label,
                                                        random_state=7,
                                                        shuffle=True,
                                                        test_size=0.2)

    X_features = null_df.drop(labels=label, axis=1)

    return [X_features, X_test, X_train, y_test, y_train]



def add_pred(grid_search, X_features, label='primary_merchant_name'):
    predictions = grid_search.predict(X_features)

    pred_df = X_features
    pred_df[label] = predictions
    
    return pred_df, predictions

def db_insert_section(section=1):

    connection = create_connection(db_name=acc.YDB_name,
                                    db_user=acc.YDB_user,
                                    db_password=acc.YDB_password,
                                    db_host=acc.YDB_host,
                                    db_port=acc.YDB_port)


    fields = ['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id', 
              'amount', 'currency', 'description', 'transaction_base_type',
              'transaction_category_name', 'primary_merchant_name', 'city',
              'state', 'transaction_origin', 'optimized_transaction_date', 
              'account_type', 'account_source_type', 'account_score', 'user_score', 
              'panel_file_created_date']


    try:
        filter_query = f"(SELECT {', '.join(field for field in fields)} FROM card_record \
                        WHERE unique_mem_id IN (SELECT unique_mem_id FROM user_demographic\
                        ORDER BY unique_mem_id ASC LIMIT 10000 OFFSET\
                        {10000*(section-1)})) UNION ALL (SELECT {', '.join(field for field in fields)} \
                         FROM bank_record WHERE unique_mem_id IN (SELECT unique_mem_id FROM \
                          user_demographic ORDER BY unique_mem_id ASC LIMIT 10000 OFFSET {10000*(section-1)}))"
        transaction_query = execute_read_query(connection, filter_query)
        main_df = pd.DataFrame(transaction_query, columns=fields)
        print(f"{len(main_df)} transactions.")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback

    for num, user in enumerate(main_df.groupby('unique_mem_id')):
        print(f"user {user[0]}, {num+1}/10000 users, {round(((num+1)/10000)*100, 2)}%.")

        # running functions in order
        df, df_null, embedding_map = df_encoder(df=main_df,
                                                sel_feature='primary_merchant_name')

        X_features, X_test, X_train, y_test, y_train = split_data(full_df=df,
                                                                  null_df=df_null)

        grid_search_svc = knn_class(X_train, X_test, y_train, y_test)

        prediction_df, predictions = add_pred(grid_search=grid_search_svc,
                                 X_features=X_features, label='primary_merchant_name')

        merchants = []
        pred_found = 0
        pred_missing = 0
        
        # for i in predictions:
        #     for val, enc in embedding_map.items():
        #         if enc in predictions:
        #             merchants.append(val)
        #         else:
        #             merchants.append("unseen in training")
        # merchants_column = merchants[:len(predictions)]
        
        # # attach merchants to prediction_df; list will be turned into df column
        # prediction_df = prediction_df.assign(merchants_pred=merchants_column)
        
        for i in predictions:
            for val, enc in embedding_map.items():
                if enc in predictions:
                    merchants.append(val)
                    pred_found += 1
                else:
                    merchants.append("unseen in training")
                    pred_missing += 1
        merchants_column = merchants[:len(X_features)]

        pr_f = round(pred_found / len(merchants) * 100, ndigits=2)
        pr_m = round(pred_missing / len(merchants) * 100, ndigits=2)
        print(f"User: {user[0]}")
        print(f"Merchants successfully decoded: {pr_f}%")
        print(f"Merchants not found in training: {pr_m}%")
        
        # attach merchants to prediction_X_features; list will be turned into df column
        feat_df = X_features.assign(merchants_pred=merchants_column)
        
        card_ess = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts']
        card_non_ess = ['Groceries', 'Automotive/Fuel', 'Home Improvement', 'Travel',
                    'Restaurants', 'Healthcare/Medical', 'Credit Card Payments',
                    'Electronics/General Merchandise', 'Entertainment/Recreation',
                    'Postage/Shipping', 'Other Expenses', 'Personal/Family',
                    'Service Charges/Fees', 'Services/Supplies', 'Utilities',
                    'Office Expenses', 'Cable/Satellite/Telecom',
                    'Subscriptions/Renewals', 'Insurance']
        bank_ess = ['Deposits', 'Salary/Regular Income', 'Transfers',
                    'Investment/Retirement Income', 'Rewards', 'Other Income',
                    'Refunds/Adjustments', 'Interest Income', 'Gifts',
                    'Expense Reimbursement']
        bank_non_ess = ['Service Charges/Fees','Credit Card Payments', 'Utilities',
                        'Healthcare/Medical', 'Loans', 'Check Payment',
                        'Electronics/General Merchandise', 'Groceries',
                        'Automotive/Fuel', 'Restaurants', 'Personal/Family',
                        'Entertainment/Recreation', 'Services/Supplies',
                        'Other Expenses', 'ATM/Cash Withdrawals',
                        'Cable/Satellite/Telecom', 'Postage/Shipping', 'Insurance',
                        'Travel', 'Taxes', 'Home Improvement', 'Education',
                        'Charitable Giving', 'Subscriptions/Renewals', 'Rent',
                        'Office Expenses', 'Mortgage']

        # Iterate through rows and mark transactions
        try:
            tr_class = pd.Series([], dtype = 'object')
            for index, i in enumerate(feat_df['transaction_category_name']):
                if i in bank_ess:
                    tr_class[index] = "essential"
                elif i in card_ess:
                    tr_class[index] = "essential"
                elif i in bank_non_ess:
                    tr_class[index] = "non_essential"
                elif i in card_non_ess:
                    tr_class[index] = "non_essential"
                else:
                    tr_class[index] = "unknown"
            feat_df = feat_df.assign(tr_essentiality=tr_class.values)
        except BaseException as error:
            print(f"column is already existing or following: {error}")

        cash_env = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Interest Income',
                    'Restaurants', 'Electronics/General Merchandise', 'Entertainment/Recreation',
                    'Postage/Shipping', 'Other Expenses', 'Other Income', 'Expense Reimbursement',
                    'Personal/Family', 'Travel', 'Office Expenses', 'Deposits',
                    'Salary/Regular Income', 'Investment/Retirement Income',
                    'ATM/Cash Withdrawals']

        bill_env = ['Check Payment', 'Rent', 'Mortgage', 'Subscriptions/Renewals',
                    'Healthcare/Medical', 'Credit Card Payments', 'Service Charges/Fees',
                    'Services/Supplies', 'Utilities', 'Insurance', 'Taxes',
                    'Home Improvement', 'Cable/Satellite/Telecom']

        car_env = ['Automotive/Fuel']
        
        gro_env = ['Groceries']
        
        chari_env = ['Charitable Giving', 'Gifts']
        
        #iterate through rows and create a new columns with a note that it is either an expense or income
        try:
            envel_cat = pd.Series([], dtype = 'object')
            for index, i in enumerate(feat_df['transaction_category_name']):
                if i in cash_env:
                    envel_cat[index] = "cash"
                elif i in bill_env:
                    envel_cat[index] = "bill"
                elif i in car_env:
                    envel_cat[index] = "automobile"
                elif i in gro_env:
                    envel_cat[index] = "groceries"
                elif i in chari_env:
                    envel_cat[index] = "charity"
                else:
                    envel_cat[index] = "not_classified"
            feat_df = feat_df.assign(envelope_category=envel_cat.values)
        except BaseException as error:
            print(f"envelope column is already existing or following {error}")

        db_name = "postgres"
        db_user = "envel"
        db_pw = "envel"
        db_host = "0.0.0.0"
        db_port = "5432"
        
        '''
        Always use %s placeholder for queries; psycopg2 will convert most data automatically
        For special cases or conversion problems use adapters or "AsIs"
        Enclose the tuples in square brackets or leave without square brackets (no performance diff)
        Dataframes have to be split up into tuples and passed as list
        '''

        try:
            connection = create_connection(db_name=db_name,
                                            db_user=db_user,
                                            db_password=db_pw,
                                            db_host=db_host,
                                            db_port=db_port)
            print("Sesson_1\n-------------")
            cursor = connection.cursor()
            sql_insert_query = """
            INSERT INTO test (test_col, test_col_2, test_col_3, test_col_4, test_col_5,
                              test_col_6, test_col_7, test_col_8, test_col_9, test_col_10,
                              test_col_11, test_col_12, test_col_13, test_col_14,
                              test_col_15, test_col_16, test_col_17, test_col_18,
                              test_col_19, test_col_20)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s);
            """

            for i in feat_df.itertuples():
            # executemany() to insert multiple rows rows
            # one-element-tuple with (i, )
                cursor.execute(sql_insert_query, ([i.unique_mem_id, i.unique_bank_account_id,
                                                   i.unique_bank_transaction_id, i.amount,
                                                   i.description, i.transaction_base_type,
                                                   i.transaction_category_name, 
                                                   i.city, i.state, i.transaction_origin,
                                                   i.optimized_transaction_date, 
                                                   i.account_type, i.account_source_type,
                                                   i.account_score, i.user_score, 
                                                   i.panel_file_created_date,
                                                   i.primary_merchant_name, i.merchants,
                                                   i.tr_essentiality, i.envelope_category]))
        
            connection.commit()
            print(len(X_features), "record(s) inserted successfully.")
        
        
        except (Exception, psycopg2.Error) as error:
            print("Failed inserting record; {}".format(error))
        
        finally:
            # closing database connection.
            if (connection):
                cursor.close()
                connection.close()
                print("Operation completed.\nPostgreSQL connection is closed.")
        print("=========================")
        return'insert done'
#%%
# running functions in order
db_insert_section(section=1)
