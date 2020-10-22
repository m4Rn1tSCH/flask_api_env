#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:09:37 2020

@author: bill
"""

'''
Purpose of this script is to interact with the dataframe and produce a new column
that either categories transactions to be part of bill or cash
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

def envelope_cat(df):


    '''
    POSTGRE-SQL COLUMNS - ALLOCATION TO ENVELOPES
    This section adds a classification of transaction categories to allow a proper allocation to either the cash or the bills envelope
    Bills describe as of 3/26/2020 all kinds of payment whose occurrence are beyond one's control,
    that come due and for which non-compliance has severe consequences
    All other kinds of payments that are of optional nature and can be avoided are classifed as cash
    '''
    
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
        for index, i in enumerate(df['transaction_category_name']):
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
                envel_cat[index] = "NOT_CLASSIFIED"
    except BaseException as error:
        print("envelope column is already existing or following {error}")
    
    
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
        INSERT INTO test (test_col_13)
        VALUES (%s);
        """
    
        # tuple or list works
        tup_ec = tuple(envel_cat)
        for i in tup_ec:
    
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

    return 'envelope categotization complete'
