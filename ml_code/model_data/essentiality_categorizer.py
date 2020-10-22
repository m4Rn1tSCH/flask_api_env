    # -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:01:06 2020

@author: bill-
"""

'''
Purpose of this script is to interact with the dataframe and produce a new column
that either marks transactions as essential or non-essential
'''
#load needed packages
import pandas as pd

def categorization(df):


    '''
    POSTGRESQL COLUMNS - CLASSIFICATION OF TRANSACTIONS
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
            if i in bank_ess or in card_ess:
                tr_class[index] = "essential"
            elif i in bank_non_ess or in card_non_ess:
                tr_class[index] = "non_essential"
            else:
                tr_class[index] = "unknown"
        df = df.assign(essentiality=tr_class.values)
    except BaseException as error:
        print("column is already existing or following: {error}")

    return df