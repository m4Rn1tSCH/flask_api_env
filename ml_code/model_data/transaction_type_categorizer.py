# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:47:31 2020

@author: bill-
"""
"""
Income and expense categorizer
"""
import pandas as pd
from datetime import datetime as dt

#%%
def type_categorizer(df):

    '''
    df: Pandas dataframe.
    returns: dataframe
    '''

    # create datetime columns
    for col in list(df):
        if df[col].dtype == 'datetime64[ns]':
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_week"] = df[col].dt.week
            df[f"{col}_weekday"] = df[col].dt.weekday

        '''
        POSTGRESQL COLUMNS - CLASSIFICATION OF TRANSACTIONS
        Following lists contains the categories to classify transactions either as expense or income
        names taken directly from the Yodlee dataset; can be appended at will
        '''
        # append these unique to DFs measuring expenses or income with their respective categories
    card_inc = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts']
    card_exp = ['Groceries', 'Automotive/Fuel', 'Home Improvement', 'Travel',
                'Restaurants', 'Healthcare/Medical', 'Credit Card Payments',
                'Electronics/General Merchandise', 'Entertainment/Recreation',
                'Postage/Shipping', 'Other Expenses', 'Personal/Family',
                'Service Charges/Fees', 'Services/Supplies', 'Utilities',
                'Office Expenses', 'Cable/Satellite/Telecom',
                'Subscriptions/Renewals', 'Insurance']

    bank_ess = ['Deposits', 'Salary/Regular Income', 'Transfers', 'Education',
                'Taxes', 'Rent', 'Mortgage', 'Healthcare/Medical', 'Investment/Retirement Income',
                'Rewards', 'Other Income', 'Utilities', 'Groceries', 'Insurance',
                'Loans', 'Cable/Satellite/Telecom', 'Refunds/Adjustments',
                'Interest Income', 'Gifts', 'Expense Reimbursement', 'Credit Card Payments',
                'Check Payment']
    bank_non_ess = ['Service Charges/Fees', 'Electronics/General Merchandise',
                    'Automotive/Fuel', 'Restaurants', 'Personal/Family',
                    'Entertainment/Recreation', 'Services/Supplies', 'Other Expenses',
                    'ATM/Cash Withdrawals', 'Postage/Shipping', 'Travel', 'Home Improvement',
                    'Charitable Giving', 'Subscriptions/Renewals', 'Office Expenses']
        #DF_CARD

        # transaction_class_card = pd.Series([], dtype = 'object')
        # for index, i in enumerate(df_card['transaction_category_name']):
        #     if i in card_inc:
        #         transaction_class_card[index] = "income"
        #     elif i in card_exp:
        #         transaction_class_card[index] = "expense"
        #     else:
        #         transaction_class_card[index] = "NOT_CLASSIFIED"
        # df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)

        #DF_BANK

    try:
        transaction_class = pd.Series([], dtype = 'object')
        for index, i in enumerate(df['transaction_category_name']):
            if i in bank_inc:
                transaction_class[index] = "income"
            elif i in bank_exp:
                transaction_class[index] = "expense"
            else:
                transaction_class[index] = "NOT_CLASSIFIED"

        df = df.assign(transaction_class=transaction_class.values)
        #except:
            #print("column is already existing or another error")
    return df

