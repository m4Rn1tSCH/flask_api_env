#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:24:04 2020

@author: bill
"""

""" Pull raw data from SQL database"""

from psycopg2 import OperationalError
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from ml_code.model_data.SQL_connection import create_connection, execute_read_query
import ml_code.model_data.PostgreSQL_credentials as acc
from ml_code.model_data.spending_report_csv_function import spending_report as create_spending_report

def pull_df(state, spending_report=False, plots=False):

    '''
    Parameters
    ----------
    rng : int, Random Seed for user picker. The default is 4.
    spending_report : bool, Save a spending report in directory if True. Default is False.
    plots : bool, Plots various graphs if True. Default is False.
    Returns
    -------
    df.
    '''

    connection = create_connection(db_name=acc.YDB_name,
                                   db_user=acc.YDB_user,
                                   db_password=acc.YDB_password,
                                   db_host=acc.YDB_host,
                                   db_port=acc.YDB_port)

    # establish connection to get users IDs per state
    try:
        filter_query = f"SELECT * FROM card_record WHERE state = '{state}'"
        transaction_query = execute_read_query(connection, filter_query)
        df = pd.DataFrame(transaction_query,
                            columns=['unique_mem_id', 'unique_bank_account_id', 'unique_bank_transaction_id',
                            'amount', 'currency', 'description', 'transaction_date', 'post_date', 'transaction_base_type',
                            'transaction_category_name', 'primary_merchant_name', 'secondary_merchant_name', 'city',
                            'state', 'zip_code', 'transaction_origin', 'factual_category', 'factual_id', 'file_created_date',
                            'optimized_transaction_date', 'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred',
                            'swipe_date', 'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
                            'account_type', 'account_source_type', 'account_score', 'user_score', 'lag', 'is_duplicate'])
        print(f"In {state} - {len(df)} transactions on record.")
        # all these columns are empty or almost empty and contain no viable information
        df = df.drop(columns=['secondary_merchant_name', 'swipe_date', 'update_type', 'is_outlier',
                              'is_duplicate', 'change_source', 'lag', 'mcc_inferred', 'mcc_raw',
                              'factual_id', 'factual_category', 'zip_code', 'yodlee_transaction_status',
                              'file_created_date', 'panel_file_created_date', 'account_source_type',
                              'account_type', 'account_score', 'user_score', 'post_date', 'transaction_date'], axis=1)
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
        ax.title("Transaction categories")
        ax.legend(loc='center right')
        plt.show()

    '''
    Generate a spending report of the unaltered dataframe
    Use the datetime columns just defined
    This report measures either the sum or mean of transactions happening
    on various days of the week/or wihtin a week or a month over the course of the year
    '''

    df['optimized_transaction_date'] = pd.to_datetime(
        df['optimized_transaction_date'])

    # set optimized transaction_date as index for later
    df.set_index('optimized_transaction_date', drop=False, inplace=True)

    df = df.drop(['unique_bank_account_id',
                  'unique_bank_transaction_id'], axis=1)
    # generate the spending report with the above randomly picked user ID
    if spending_report:
        create_spending_report(df=df.copy())

    return df
