'''
This module conducts the filtering of dataframes per unique user ID
Following steps are involved:
    adding feature column:"income/expense"
    summing up values per user and transaction class
    calculates injection needed before first income is received
'''
import numpy as np
import pandas as pd
import csv

#as a self function to use as method
def injector(self):
    #test_path = r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx'
    #relative path to test the file sitting directly in the folder with the script
    #test_path_2 = './2020-01-28 envel.ai Working Class Sample.xlsx'

    #df_card = pd.read_excel(os.path.abspath(test_path), sheet_name = "Card Panel")
    #card_members = df_card['unique_mem_id'].unique()
    #%%
    '''
    POSTGRESQL COLUMNS - CLASSIFICATION OF TRANSACTIONS
    Following lists contains the categories to classify transactions either as expense or income
    names taken directly from the Yodlee dataset; can be appended at will
    '''
    #append these unique to DFs measuring expenses or income with their respective categories
    card_inc = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts']
    card_exp = ['Groceries', 'Automotive/Fuel', 'Home Improvement', 'Travel',
                'Restaurants', 'Healthcare/Medical', 'Credit Card Payments',
                'Electronics/General Merchandise', 'Entertainment/Recreation',
                'Postage/Shipping', 'Other Expenses', 'Personal/Family',
                'Service Charges/Fees', 'Services/Supplies', 'Utilities',
                'Office Expenses', 'Cable/Satellite/Telecom',
                'Subscriptions/Renewals', 'Insurance']
    bank_inc = ['Deposits', 'Salary/Regular Income', 'Transfers',
                'Investment/Retirement Income', 'Rewards', 'Other Income',
                'Refunds/Adjustments', 'Interest Income', 'Gifts', 'Expense Reimbursement']
    bank_exp = ['Service Charges/Fees',
                'Credit Card Payments', 'Utilities', 'Healthcare/Medical', 'Loans',
                'Check Payment', 'Electronics/General Merchandise', 'Groceries',
                'Automotive/Fuel', 'Restaurants', 'Personal/Family',
                'Entertainment/Recreation', 'Services/Supplies', 'Other Expenses',
                'ATM/Cash Withdrawals', 'Cable/Satellite/Telecom',
                'Postage/Shipping', 'Insurance', 'Travel', 'Taxes',
                'Home Improvement', 'Education', 'Charitable Giving',
                'Subscriptions/Renewals', 'Rent', 'Office Expenses', 'Mortgage']
    '''
    Iterate through rows and create new columns with a keyword that it is either an expense or income
    This part is needed to make sure that initial balances can be determined better
    '''
    #DF_CARD
    #try:
    transaction_class_card = pd.Series([], dtype = 'object')
    for index, i in enumerate(self['transaction_category_name']):
        if i in card_inc:
            transaction_class_card[index] = "income"
        elif i in card_exp:
            transaction_class_card[index] = "expense"
        else:
            transaction_class_card[index] = "NOT_CLASSIFIED"
    self.insert(loc = len(self.columns), column = "transaction_class", value = transaction_class_card)
    #except:
        #print("column is already existing or another error")
    ###################################
    #DF_BANK
    #try:
    # transaction_class_bank = pd.Series([], dtype = 'object')
    # for index, i in enumerate(df_bank['transaction_category_name']):
    #     if i in bank_inc:
    #         transaction_class_bank[index] = "income"
    #     elif i in bank_exp:
    #         transaction_class_bank[index] = "expense"
    #     else:
    #         transaction_class_bank[index] = "NOT_CLASSIFIED"
    # df_bank.insert(loc = len(df_bank.columns), column = "transaction_class", value = transaction_class_bank)
    #except:
        #print("column is already existing or another error")
    #%%
    '''
    Filter for dataframes to find out income and expenses narrowed down to the user id
    '''
    #filter with ilocation and show expenses and income as separate dataframe
    #card_expenses = self.iloc[np.where(self['transaction_class'] == "expense")]
    #card_expenses_by_user = self.iloc[np.where(self['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    #card_income = self.iloc[np.where(self['transaction_class'] == "income")]
    #card_income_by_user = self.iloc[np.where(self['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    #bank_expenses = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")]
    #bank_expenses_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    #bank_income = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")]
    #bank_income_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    #%%
    df_1 = self[['unique_mem_id', 'amount', 'transaction_class']].groupby('unique_mem_id')
    #%%
    print("CARD PANEL INJECTION")
    #open initially and only write to the file to generate the headers
    with open('Card_Panel_Injection.csv', 'w') as newFile:
        newFileWriter=csv.writer(newFile)
        newFileWriter.writerow("Refers to: CARD_PANEL")
        newFileWriter.writerow(["User_ID", "Injection in USD required"])
    # f = open('test.csv', 'w')
    # with f:
        # fnames = ["User_ID", "Injection in USD required"]
        # writer = csv.DictWriter(f, fieldnames=fnames)
        # writer.writeheader()

    #DF_1
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_1.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print(index, row.unique_mem_id, row.amount, row.transaction_class)
                amount_list.append(row.amount)
                cumulative_amount = np.cumsum(amount_list, axis = 0)
                #print(row.unique_mem_id, cumulative_amount)
            else:
                #print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
                break
        #print out the member id as part of the for-loop and and the last element of the list which is the amount to be injected
        print(f"unique_member_ID: {row.unique_mem_id}; initial injection needed in USD: {cumulative_amount[-1]}")

        #open/append income and expense per user_id to a CSV that has been created outside the loop
        #writes all rows inside the iteration loop correctly but without headers now
        with open('Card_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, cumulative_amount[-1]])
        ##f = open('test.csv', 'a')
        #with f:
            #field names needed in append mode to know the orders of keys and values
            #fnames = ['User_ID', 'Injection in USD required']
            #writer = csv.DictWriter(f, fieldnames=fnames)
            #writer.writerow({'User_ID' : row.unique_mem_id, 'Injection in USD required': cumulative_amount[-1]})

    except Exception as exc:
        for row in df_1.head(1).itertuples():
            with open('Card_Panel_Injection.csv', 'a') as newFile:
                newFileWriter=csv.writer(newFile)
                #write per row to a CSV
                newFileWriter.writerow([row.unique_mem_id, exc])
            ##f = open('test.csv', 'a')
            #with f:
                #field names needed in append mode to know the orders of keys and values
                #fnames = ['User_ID', 'Injection in USD required']
                #writer = csv.DictWriter(f, fieldnames=fnames)
                #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        #print(f"There was a problem with user ID: {card_members[0]}; Error: {exc}")
        print(f"There was a problem with user ID: {row.unique_mem_id}; Error: {exc}")
        pass

    if __name__ == "__main__":
        import sys
        close_connection(int(sys.argv[1]))