#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:36:52 2020

@author: bill
"""

#for presentation
import pandas as pd
import psycopg2
from ml_code.model_data.SQL_connection import create_connection
from ml_code.model_data.dataframe_insert import df_insert_query
from ml_code.model_data.essentiality_sql_insert import essentiality_cat
from ml_code.model_data.envelope_sql_insert import envelope_cat
from ml_code.model_data.raw_data_connection import pull_df

df = pull_df(rng=97,
              spending_report=False,
              plots=False)


#df_insert_query(df=df)
#essentiality_cat(df=df)
#envelope_cat(df=df)

#################################
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
    df = df.assign(tr_essentiality=tr_class.values)
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
    df = df.assign(envelope_category=envel_cat.values)
except BaseException as error:
    print(f"envelope column is already existing or following {error}")


################################

db_name = "postgres"
db_user = "envel"
db_pw = "envel"
db_host = "0.0.0.0"
db_port = "5432"

'''
Always use %s placeholder for queries; psycopg2 will convert most data automatically
For special cases or conversion problems use adapters or "AsIs"
Enclose the tuples in square brackets or leave without square brackets (no performance diff)
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
    INSERT INTO test (test_col, test_col_2, test_col_3, test_col_4,
                      test_col_5, test_col_6, test_col_7, test_col_8,
                      test_col_9, test_col_10, test_col_11, test_col_12,
                      test_col_13)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    # merch_list = np.ndarray(['Tatte Bakery', 'Star Market', 'Stop n Shop', 'Auto Parts Shop',
    #               'Trader Joes', 'Insomnia Cookies'])

    # tuple or list works
    # split up into tuples and pass as list
    for i in df.itertuples():
    # executemany() to insert multiple rows rows
    # one-element-tuple with (i, )
        cursor.execute(sql_insert_query, ([i.unique_mem_id, i.amount, i.currency,
                                           i.description, i.transaction_base_type,
                                           i.transaction_category_name,
                                           i.primary_merchant_name, i.city,
                                           i.state, i.transaction_origin,
                                           i.optimized_transaction_date,
                                           i.tr_essentiality,
                                           i.envelope_category]))

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
