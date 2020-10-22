#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:05:30 2020

@author: bill
"""

'''
Purpose of this script is to interact with the dataframe and produce a new column
that either marks transactions as essential or non-essential
'''
#load needed packages
import pandas as pd
from ml_code.model_data.SQL_connection import create_connection
from ml_code.model_data.raw_data_connection import pull_df
import psycopg2

#TEST_DF
# df = pull_df(rng=22,
#              spending_report=False,
#              plots=False)
#########################

def essentiality_cat(df):


    '''
    POSTGRESQL COLUMN - CLASSIFICATION OF TRANSACTIONS
    Following lists contains the categories to classify transactions either as expense or income
    names taken directly from the Yodlee dataset; can be appended at will
    '''
    #essential/ non-essential transactions
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
        for index, i in enumerate(df['transaction_category_name']):
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
    except BaseException as error:
        print("column is already existing or following: {error}")


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
        print("-------------")
        cursor = connection.cursor()
        sql_insert_query = """
        INSERT INTO test (test_col_12)
        VALUES (%s);
        """
    
        # tuple or list works
        tup_tr = tuple(tr_class)
        for i in tup_tr:
    
        # one-element-tuple with (i, )
            cursor.execute(sql_insert_query, (i, ))
    
        connection.commit()
        print(len(df), "record(s) inserted successfully.")
    
    except (Exception, psycopg2.Error) as error:
        print("Failed inserting record; {}".format(error))
    
    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("Operation completed.\nPostgreSQL connection is closed.")
    print("=========================")

    return 'essentiality insert complete'
