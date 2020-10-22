'''
Purpose of this script is to interact with a csv file/database and produce a dictionary
with unique IDs and corresponding income and expenses in separate dictionaries
-determine income and expenses based on given categories
-add it either to the INCOME DF or the EXPENSE DF and export it later
-find out daily, weekly and monthly throughput of accounts and their excess cash
-develop a logic for daily limits and spending patterns
'''
import pandas as pd
import numpy as np
from datetime import datetime as dt
from flask import Flask
import os
import csv
#%%
    #CONNECTION TO FLASK/SQL
app = Flask(__name__)

##put address here
#function can be bound to the script ny adding a new URL
#e.g. route('/start') would then start the entire function that follows
#same can be split up
@app.route('/start')

    ########SETTING THE ENVIRONMENT VARIABLE#######
    #$ set FLASK_APP=file_name.py
    #$ flask run
    # * Running on http://127.0.0.1:5000/

    ####COMMAND PROMPT#####
    #set env var in windows with name: FLASK_APP and value: path to app.py
    #switch in dev consolde to this path
    #C:\path\to\app>set FLASK_APP=hello.py
    ####for production use##
    #make the flask app listen to all public IPs
    ##ONLY FOR TESTING; MAJOR SECURITY ISSUE
    #flask run --host=0.0.0.0

    ##joint command to set env var and run the app
    #env FLASK_APP=Python_inc_exp_bal_database.py flask run

##remove paths outside of functions before running it with flask

def preproccessing(file_path):
    '''
    REPLACE THE TEST_PATH HERE IF YOU RUN THE FUNCTION EXTERNALLY
    '''
    test_path = r'C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx'
    df_card = pd.read_excel(os.path.abspath(test_path), sheet_name = "Card Panel")
    df_bank = pd.read_excel(os.path.abspath(test_path), sheet_name = "Bank Panel")
    df_demo = pd.read_excel(os.path.abspath(test_path), sheet_name = "User Demographics")
    #in the non-encoded verion all columns still have correct types
    #extract unique numbers from all panels to find out unique users;
    card_members = df_card['unique_mem_id'].unique()
    bank_members = df_bank['unique_mem_id'].unique()
    demo_members = df_demo['unique_mem_id'].unique()
    trans_cat_card = df_card['transaction_category_name'].unique()
    trans_cat_bank = df_bank['transaction_category_name'].unique()
    #%%
    '''
    Brief check if all customers given in the demographics panel are also having transactions in the card or bank panel
    Also to check if some customers are not even listed in the demo panel
    ambiguous values in arrays require .all() /.any() for comparison
    '''
    if card_members.all() == demo_members.all():
        print("In card panel: demo users are identical to card panel users")
    else:
        print("In card panel: users dissimilar!")
    if bank_members.all() == demo_members.all():
        print("In bank panel: demo users are identical to bank panel users")
    else:
        print("In bank panel: users dissimilar!")
    #%%
    '''
    Datetime engineering for card and bank panel
    These columns help for reporting like weekly or monthly expenses and
    improve prediction of re-occurring transactions
    '''
    for col in list(df_card):
        if df_card[col].dtype == 'datetime64[ns]':
            df_card[f"{col}_month"] = df_card[col].dt.month
            df_card[f"{col}_week"] = df_card[col].dt.week
            df_card[f"{col}_weekday"] = df_card[col].dt.weekday
    #%%
    #Datetime engineering DF_BANK
    for col in list(df_bank):
        if df_bank[col].dtype == 'datetime64[ns]':
            df_bank[f"{col}_month"] = df_bank[col].dt.month
            df_bank[f"{col}_week"] = df_bank[col].dt.week
            df_bank[f"{col}_weekday"] = df_bank[col].dt.weekday
    #%%
    #DATETIME ENGINEERING
    #this includes expenses and income
    #mean + stdev of past 3d/7d/30d/ + rolling volume
    df_card.reset_index(drop = True, inplace = True)
    #pick lag features to iterate through and calculate features
    #original lag features; based on tutorial dataset
    #lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
    lag_features = ["amount"]
    #set up time frames; how many days/months back/forth
    t1 = 3
    t2 = 7
    t3 = 30
    #rolling values for all columns ready to be processed
    #DataFrame.rolling(self, window, min_periods = None, center = False, win_type = None, on = None, axis = 0, closed = None)
    #DataFrame.shift(self, periods = 1, freq = None, axis = 0, fill_value = None)
    df_card_rolled_3d = df_card[lag_features].rolling(window = t1, min_periods = 0)
    df_card_rolled_7d = df_card[lag_features].rolling(window = t2, min_periods = 0)
    df_card_rolled_30d = df_card[lag_features].rolling(window = t3, min_periods = 0)

    #calculate the mean with a shifting time window
    df_card_mean_3d = df_card_rolled_3d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_card_mean_7d = df_card_rolled_7d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_card_mean_30d = df_card_rolled_30d.mean().shift(periods = 1).reset_index().astype(np.float32)

    #calculate the std dev with a shifting time window
    df_card_std_3d = df_card_rolled_3d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_card_std_7d = df_card_rolled_7d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_card_std_30d = df_card_rolled_30d.std().shift(periods = 1).reset_index().astype(np.float32)

    for feature in lag_features:
        df_card[f"{feature}_mean_lag{t1}"] = df_card_mean_3d[feature]
        df_card[f"{feature}_mean_lag{t2}"] = df_card_mean_7d[feature]
        df_card[f"{feature}_mean_lag{t3}"] = df_card_mean_30d[feature]

        df_card[f"{feature}_std_lag{t1}"] = df_card_std_3d[feature]
        df_card[f"{feature}_std_lag{t2}"] = df_card_std_7d[feature]
        df_card[f"{feature}_std_lag{t3}"] = df_card_std_30d[feature]

    #fill missing values with the mean to keep distortion very low and allow prediction
    df_card.fillna(df_card.mean(), inplace = True)
    #associate date as the index columns to columns (especially the newly generated ones to allow navigating and slicing)
    #df_card.set_index("transaction_date", drop = False, inplace = True)
    #%%
    #DATETIME ENGINEERING
    df_bank.reset_index(drop = True, inplace = True)
    #pick lag features to iterate through and calculate features
    #original lag features; based on tutorial dataset
    #lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
    lag_features = ["amount"]
    #set up time frames; how many days/months back/forth
    t1 = 3
    t2 = 7
    t3 = 30
    #rolling values for all columns ready to be processed
    #DataFrame.rolling(self, window, min_periods = None, center = False, win_type = None, on = None, axis = 0, closed = None)
    #rolling method; window = size of the moving window;
                    #min_periods = min no. of obersvations that need to have a value(otherwise result is NA)
                    #center = set labels at the center of the window
                    #win_type = weighting of points, "None" all points are equally weighted
                    #on = use datetime-like column index (instead of df indices) to calculate the value
                    #axis = 0:row-wise; 1:column-wise
                    #closed = ['right', 'left', 'both', 'neither'] close of the interval; for offset-based windows defaults to rights;
                    #for fixed windows defaults to both
    #DataFrame.shift(self, periods = 1, freq = None, axis = 0, fill_value = None)
                    #periods = pos/ neg downwards or upwards shift in periods
                    #freq = offset/timedelta/str; index shifted but data not realigned; extend index when shifting + preserve original data
                    #axis = shift direction (0: index 1: columns None)
                    #fill_value = numeric: np.nan; datetime,timedelta: NaT; extension types:dtype.na_value
    df_bank_rolled_3d = df_bank[lag_features].rolling(window = t1, min_periods = 0)
    df_bank_rolled_7d = df_bank[lag_features].rolling(window = t2, min_periods = 0)
    df_bank_rolled_30d = df_bank[lag_features].rolling(window = t3, min_periods = 0)

    #calculate the mean with a shifting time window
    df_bank_mean_3d = df_bank_rolled_3d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_mean_7d = df_bank_rolled_7d.mean().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_mean_30d = df_bank_rolled_30d.mean().shift(periods = 1).reset_index().astype(np.float32)

    #calculate the std dev with a shifting time window
    df_bank_std_3d = df_bank_rolled_3d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_std_7d = df_bank_rolled_7d.std().shift(periods = 1).reset_index().astype(np.float32)
    df_bank_std_30d = df_bank_rolled_30d.std().shift(periods = 1).reset_index().astype(np.float32)

    for feature in lag_features:
        df_bank[f"{feature}_mean_lag{t1}"] = df_bank_mean_3d[feature]
        df_bank[f"{feature}_mean_lag{t2}"] = df_bank_mean_7d[feature]
        df_bank[f"{feature}_mean_lag{t3}"] = df_bank_mean_30d[feature]

        df_bank[f"{feature}_std_lag{t1}"] = df_bank_std_3d[feature]
        df_bank[f"{feature}_std_lag{t2}"] = df_bank_std_7d[feature]
        df_bank[f"{feature}_std_lag{t3}"] = df_bank_std_30d[feature]

    #fill missing values with the mean to keep distortion very low and allow prediction
    df_bank.fillna(df_bank.mean(), inplace = True)
    #associate date as the index columns to columns (especially the newly generated ones to allow navigating and slicing)
    #df_bank.set_index("transaction_date", drop = False, inplace = True)
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
    for index, i in enumerate(df_card['transaction_category_name']):
        if i in card_inc:
            transaction_class_card[index] = "income"
        elif i in card_exp:
            transaction_class_card[index] = "expense"
        else:
            transaction_class_card[index] = "NOT_CLASSIFIED"
    df_card.insert(loc = len(df_card.columns), column = "transaction_class", value = transaction_class_card)
    #except:
        #print("column is already existing or another error")
    ###################################
    #DF_BANK
    #try:
    transaction_class_bank = pd.Series([], dtype = 'object')
    for index, i in enumerate(df_bank['transaction_category_name']):
        if i in bank_inc:
            transaction_class_bank[index] = "income"
        elif i in bank_exp:
            transaction_class_bank[index] = "expense"
        else:
            transaction_class_bank[index] = "NOT_CLASSIFIED"
    df_bank.insert(loc = len(df_bank.columns), column = "transaction_class", value = transaction_class_bank)
    #except:
        #print("column is already existing or another error")
#%%
    '''
    POSTGRE-SQL COLUMNS - ALLOCATION TO ENVELOPES
    This section adds a classification of transaction categories to allow a proper allocation to either the cash or the bills envelope
    Bills describes as of 3/26/2020 all kinds of payment whose occurrence is beyond one's control,
    that comes due and for which non-compliance has evere consequences
    All other kinds of payments that are of optional nature and can be avoided are classifed as cash
    '''
    cash_env_card = ['Rewards', 'Transfers', 'Refunds/Adjustments', 'Gifts',
                     'Restaurants', 'Electronics/General Merchandise',
                     'Entertainment/Recreation', 'Postage/Shipping', 'Other Expenses',
                     'Personal/Family','Groceries', 'Automotive/Fuel',  'Travel']

    bill_env_card = ['Home Improvement', 'Healthcare/Medical', 'Credit Card Payments'
                     'Service Charges/Fees', 'Services/Supplies', 'Utilities',
                     'Office Expenses', 'Cable/Satellite/Telecom',
                     'Subscriptions/Renewals', 'Insurance']

    cash_env_bank = ['Deposits', 'Salary/Regular Income', 'Transfers',
                     'Investment/Retirement Income', 'Rewards', 'Other Income',
                     'Refunds/Adjustments', 'Interest Income', 'Gifts', 'Expense Reimbursement',
                     'Electronics/General Merchandise', 'Groceries', 'Automotive/Fuel',
                     'Restaurants', 'Personal/Family', 'Entertainment/Recreation',
                     'Services/Supplies', 'Other Expenses', 'ATM/Cash Withdrawals',
                     'Postage/Shipping', 'Travel', 'Education', 'Charitable Giving',
                     'Office Expenses']

    bill_env_bank = ['Service Charges/Fees', 'Credit Card Payments',
                     'Utilities', 'Healthcare/Medical', 'Loans', 'Check Payment',
                     'Cable/Satellite/Telecom', 'Insurance', 'Taxes', 'Home Improvement',
                     'Subscriptions/Renewals', 'Rent', 'Mortgage']
    #iterate through rows and create a new columns with a note that it is either an expense or income
    #DF_CARD
    #try:
    envelope_cat_card = pd.Series([], dtype = 'object')
    for index, i in enumerate(df_card['transaction_category_name']):
        if i in cash_env_card:
            envelope_cat_card[index] = "cash"
        elif i in bill_env_card:
            envelope_cat_card[index] = "bill"
        else:
            envelope_cat_card[index] = "NOT_CLASSIFIED"
    df_card.insert(loc = len(df_card.columns), column = "envelope_category", value = envelope_cat_card)
    #except:
        #print("CASH/BILL column is already existing or another error")
    ##############################
    #DF_BANK
    #try:
    envelope_cat_bank = pd.Series([], dtype = 'object')
    for index, i in enumerate(df_bank['transaction_category_name']):
        if i in cash_env_bank:
            envelope_cat_bank[index] = "cash"
        elif i in bill_env_bank:
            envelope_cat_bank[index] = "bill"
        else:
            envelope_cat_bank[index] = "NOT_CLASSIFIED"
    df_bank.insert(loc = len(df_bank.columns), column = "envelope_category", value = envelope_cat_bank)
    #except:
        #print("CASH/BILL column is already existing or another error")
    #%%
    '''
    Filter for dataframes to find out income and expenses narrowed down to the user id
    '''
    #filter with ilocation and show expenses and income as spearate dataframe
    card_expenses = df_card.iloc[np.where(df_card['transaction_class'] == "expense")]
    card_expenses_by_user = df_card.iloc[np.where(df_card['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    card_income = df_card.iloc[np.where(df_card['transaction_class'] == "income")]
    card_income_by_user = df_card.iloc[np.where(df_card['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    bank_expenses = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")]
    bank_expenses_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "expense")].groupby('unique_mem_id').sum()
    bank_income = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")]
    bank_income_by_user = df_bank.iloc[np.where(df_bank['transaction_class'] == "income")].groupby('unique_mem_id').sum()
    #%%
    '''
    POSTGRESQL - BUDGET SUGGESTION COLUMNS
    Create columns with an initial recommendation of the budgeting mode and the corresponding daily limit
    Logic is based on the weekly or biweekly income:
    Logic of stability of spending behavior and standard deviation within various time frames
    Behavior is considered: stable and non-erratic when:
        LATER:Std dev of past 3 days is still smaller than emergency cash allocated for a day
        LATER:Std dev of past week is still smaller than emergency allocated for a week
        LATER:Std dev of 30d is smaller than 70% of monthly income
        (to allow purchase of flight tickets or hotel stays without forcing a change of the spending mode)
    '''
    #DF_CARD
    #try:
    print("CARD PANEL BUDGETING REPORT")
    budget_mode_card = pd.Series([], dtype = 'object')
    for index, i, e, c in zip(bank_income_by_user.index, bank_income_by_user.amount,
                              bank_expenses_by_user.amount, card_expenses_by_user.amount):
        if i > e + c:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Good Budget!")
            budget_mode_card[index] = "normal mode"
        else:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Overspending!")
            budget_mode_card[index] = "beastmode"
    df_card.insert(loc = len(df_card.columns), column = "budget_mode_suggestion_card", value = budget_mode_card)
    #except:
        #print("values overwritten in card panel")
    #DF_BANK
    #try:
    budget_mode_bank = pd.Series([], dtype = 'object')
    print("BANK PANEL BUDGETING REPORT")
    for index, i, e, c in zip(bank_income_by_user.index, bank_income_by_user.amount,
                              bank_expenses_by_user.amount, card_expenses_by_user.amount):
        if i > e + c:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Good Budget!")
            budget_mode_bank[index] = "normal mode"
        else:
            print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Overspending!")
            budget_mode_bank[index] = "beastmode"
    df_bank.insert(loc = len(df_bank.columns), column = "budget_mode_suggestion_card", value = budget_mode_bank)
    #except:
        #print("values overwritten in bank panel")
    #%%
    df_card.set_index("optimized_transaction_date", drop = False, inplace = True)
    df_bank.set_index("optimized_transaction_date", drop = False, inplace = True)
    #%%
    '''
    70850441974905670928446
    201492116860211330700059
    257154737161372702866152
    364987015290224198196263
    651451454569880463282551
    748150568877494117414131
    '''
    #%%
    '''
    IMPROVISED SOLUTION WITHOUT ITERATION
    Filter-df by unique id of each customer with columns: member_id; amount; envelope_category; transaction_class
    iteration over each row as tuples and append amount to a list.
    This list is taken and used for a cumulative sum of all transactions with type "expense"
    Until "income" class is hit to stop
    Numerical amount needs to be injected for simulation
    problem of Python here; one cannot assign an element to a list that is not yet existing
    '''
    df_1 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '70850441974905670928446']
    df_2 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '201492116860211330700059']
    df_3 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '257154737161372702866152']
    df_4 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '364987015290224198196263']
    df_5 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '651451454569880463282551']
    df_6 = df_card[['unique_mem_id', 'amount', 'transaction_class']][df_card['unique_mem_id'] == '748150568877494117414131']
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

        print(f"There was a problem with user ID: {card_members[0]}; Error: {exc}")
        pass

    ##DF_2
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_2.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[1]}; Error: {exc}")
        pass

    ##DF_3
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_3.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[2]}; Error: {exc}")
        pass

    ##DF_4
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_4.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[3]}; Error: {exc}")
        pass

    ##DF_5
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_5.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[4]}; Error: {exc}")
        pass

    ##DF_6
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_6.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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

        print(f"There was a problem with user ID: {card_members[5]}; Error: {exc}")
        pass
    #%%
    '''
    '70850441974905670928446'
    '257154737161372702866152'
    '364987015290224198196263'
    '579758724513140495207829'
    '630323465162087035360618'
    '635337295180631420039874'
    '1187627404526562698645364'
    '''
    df_70850441974905670928446 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '70850441974905670928446']
    df_257154737161372702866152 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '257154737161372702866152']
    df_364987015290224198196263 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '364987015290224198196263']
    df_579758724513140495207829 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '579758724513140495207829']
    df_630323465162087035360618 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '630323465162087035360618']
    df_635337295180631420039874 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '635337295180631420039874']
    df_1187627404526562698645364 = df_bank[['unique_mem_id', 'amount', 'transaction_class']][df_bank['unique_mem_id'] == '1187627404526562698645364']
    #%%
    print("BANK PANEL INJECTION")
    #open initially and only write to the file to generate the headers
    with open('Bank_Panel_Injection.csv', 'w') as newFile:
        newFileWriter=csv.writer(newFile)
        newFileWriter.writerow(["Refers to:", "BANK_PANEL"])
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
        for row in df_70850441974905670928446.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print(index, row.unique_mem_id, row.amount, row.transaction_class)
                amount_list.append(row.amount)
                cumulative_amount = np.cumsum(amount_list, axis = 0)
                print(row.unique_mem_id, cumulative_amount)
            else:
                #print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
                break
        #print out the member id as part of the for-loop and and the last element of the list which is the amount to be injected
        print(f"unique_member_ID: {row.unique_mem_id}; initial injection needed in USD: {cumulative_amount[-1]}")
        #open/append income and expense per user_id to a CSV that has been created outside the loop
        #writes all rows inside the iteration loop correctly but without headers now
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        ##f = open('test.csv', 'a')
        #with f:
            #field names needed in append mode to know the orders of keys and values
            #fnames = ['User_ID', 'Injection in USD required']
            #writer = csv.DictWriter(f, fieldnames=fnames)
            #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})
        print(f"There was a problem with user ID: {bank_members[0]}; Error: {exc}")
        pass

    ##DF_2
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_257154737161372702866152.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        ##f = open('test.csv', 'a')
        #with f:
            #field names needed in append mode to know the orders of keys and values
            #fnames = ['User_ID', 'Injection in USD required']
            #writer = csv.DictWriter(f, fieldnames=fnames)
            #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[1]}; Error: {exc}")
        pass

    ##DF_3
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_364987015290224198196263.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[2]}; Error: {exc}")
        pass

    ##DF_4
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_579758724513140495207829.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[3]}; Error: {exc}")
        pass

    ##DF_5
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_630323465162087035360618.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
                amount_list.append(row.amount)
                cumulative_amount = np.cumsum(amount_list, axis = 0)
                #print(row.unique_mem_id, cumulative_amount)
            else:
               # print(f"stopped at user_ID: {row.unique_mem_id}, cumulative sum injected: {cumulative_amount[-1]}")
                break
        #print out the member id as part of the for-loop and and the last element of the list which is the amount to be injected
        print(f"unique_member_ID: {row.unique_mem_id}; initial injection needed in USD: {cumulative_amount[-1]}")

        #open/append income and expense per user_id to a CSV that has been created outside the loop
        #writes all rows inside the iteration loop correctly but without headers now
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, cumulative_amount[-1]])
        ##f = open('test.csv', 'a')
        #with f:
            #field names needed in append mode to know the orders of keys and values
            #fnames = ['User_ID', 'Injection in USD required']
            #writer = csv.DictWriter(f, fieldnames=fnames)
            #writer.writerow({'User_ID' : row.unique_mem_id, 'Injection in USD required': cumulative_amount[-1]})

    except  Exception as exc:

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[4]}; Error: {exc}")
        pass

    ##DF_6
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_635337295180631420039874.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[5]}; Error: {exc}")
        pass

    ##DF_7
    try:
        cumulative_amount = []
        amount_list = []
        for row in df_1187627404526562698645364.itertuples():
            #access data using column names
            if row.transaction_class == "expense":
                #print an overview and calculate the cumulative sum
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
        with open('Bank_Panel_Injection.csv', 'a') as newFile:
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

        with open('Bank_Panel_Injection.csv', 'a') as newFile:
            newFileWriter=csv.writer(newFile)
            #write per row to a CSV
            newFileWriter.writerow([row.unique_mem_id, exc])
        # ##f = open('test.csv', 'a')
        # #with f:
        #     #field names needed in append mode to know the orders of keys and values
        #     #fnames = ['User_ID', 'Injection in USD required']
        #     #writer = csv.DictWriter(f, fieldnames=fnames)
        #     #writer.writerow({'User_ID' : "Problem with:" row.unique_mem_id, 'Injection in USD required': "Problem:" exc})

        print(f"There was a problem with user ID: {bank_members[6]}; Error: {exc}")
        pass
#%%
    '''
    CHART FOR EACH USER'S INCOME, EXPENSES AND EXCESS MONEY
    The loop uses the filtered dataframes which are narrowed down by user and
    show the budgeting ability of unique user ID found in the panel
    '''
    #index = index
    #i = income
    #e = expense
    '''
    REPORTING CSV - YODLEE DATA
    Write it on a per-line basis to the csv that will either sit sit in the flask folder
    or can be saved in the current working directory and will deliver information for the disconnected injector
    '''
    try:
        #open initially and only write to the file to generate the headers
        with open('User_ID_transactions.csv','w') as newFile:
            newFileWriter=csv.writer(newFile)
            newFileWriter.writerow(["User_ID", "Income", "Expenses", "Excess_Cash(+)/Debt(-)"])
        # f = open('test.csv', 'w')
        # with f:
            # fnames = ['User_ID', 'income', 'expense', 'difference']
            # writer = csv.DictWriter(f, fieldnames=fnames)
            # writer.writeheader()
        for index, i, e in zip(bank_income_by_user.index, bank_income_by_user.amount, bank_expenses_by_user.amount):
            if i > e:
                print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Good Budget!; Excess cash: {i - e}")
            else:
                print(f"User_ID: {index}; Income: {i}; Expenses: {e}; Overspending!; Debt: {i - e}")
            #open/append income and expense per user_id to a CSV that has been created outside the loop
            #writes all rows inside the iteration loop correctly but without headers now
            with open('User_ID_transactions.csv','a') as newFile:
                newFileWriter=csv.writer(newFile)
                #write per row to a CSV
                newFileWriter.writerow([index, i, e, i - e])
            ##f = open('test.csv', 'a')
            #with f:
                #field names needed in append mode to know the orders of keys and values
                #fnames = ['User_ID', 'income', 'expense', 'difference']
                #writer = csv.DictWriter(f, fieldnames=fnames)
                #writer.writerow({'User_ID' : index, 'income': i, 'expense': e, 'difference': i-e})
    except:
        print("data by user might not be available; check the number of unique user IDs")
    #%%
    '''
    Addition of feature columns for additive spending on a weekly; monthly; daily basis
    These dataframes are then convertable to a CSV for reporting purposes or could be shown in the app
    '''
    #total throughput of money
    total_throughput = df_card['amount'].sum()
    #monthly figures
    net_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).sum()
    avg_monthly_throughput = df_card['amount'].groupby(df_card['transaction_date_month']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    monthly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
    #monthly_expenses = df_card['amount'][df_card['transaction_base_type'] == 'debit'].groupby(df_card['transaction_date_week']).sum()
    #weekly figures
    net_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).sum()
    avg_weekly_throughput = df_card['amount'].groupby(df_card['transaction_date_week']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    weekly_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_week']).sum()
    #weekly_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_week']).sum()
    #daily figures
    net_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).sum()
    avg_daily_spending = df_card['amount'].groupby(df_card['transaction_date_weekday']).mean()
    #CHECK VIABILITY OF SUCH VARIABLES
    daily_gain = df_card['amount'][df_card['amount'] >= 0].groupby(df_card['transaction_date_weekday']).sum()
    #daily_expenses = df_card['amount'][df_card['transaction_base_type'] == "debit"].groupby(df_card['transaction_date_weekday']).sum()
    #report for users about their spending patterns, given in various intervals
    try:
        print(f"The total turnover on your account has been ${total_throughput}")
        print("................................................................")
        spending_metrics_monthly = pd.DataFrame(data = {'Average Monthly Spending':avg_monthly_throughput,
                                                        'Monthly Turnover':net_monthly_throughput})
        print(spending_metrics_monthly)
        print(".................................................................")
        spending_metrics_weekly = pd.DataFrame(data = {'Average Weekly Spending':avg_weekly_throughput,
                                                       'Weekly Turnover':net_weekly_throughput})
        print(spending_metrics_weekly)
        print(".................................................................")
        spending_metrics_daily = pd.DataFrame(data = {'Average Daily Spending':avg_daily_spending,
                                                      'Daily Turnover':net_daily_spending})
        print(spending_metrics_daily)
    except:
        print("You do not have enough transactions yet. But we are getting there...")
#%%
    '''
            CONVERSION OF THE ENTIRE DATAFRAMES
    For testing purposes which does not include randomized IDs as part of the name and allows loading a constant name
    AFTER CSV IS GENERATED(in pred_func):
        Reassert datetime objects to all date columns
        Set Transaction date as index
    The dataframe is now preprocessed and ready to be loaded by the prediction models for predictive analysis
    '''
    #Conversion of df to CSV or direct pass possible
    raw = os.getcwd()
    date_of_creation = dt.today().strftime('%m-%d-%Y_%Hh-%mmin')

    path_card = os.path.abspath(os.path.join(raw, date_of_creation + '_CARD_PANEL_PROCESSED' + '.csv'))
    path_bank = os.path.abspath(os.path.join(raw, date_of_creation + '_BANK_PANEL_PROCESSED' + '.csv'))
    #path_demo = os.path.abspath(os.path.join(raw, date_of_creation + '_DEMO_PANEL_PROCESSED' + '.csv'))
    try:
        df_card.to_csv(path_card)
        df_bank.to_csv(path_bank)
        #df_demo.to_csv(csv_path_demo)
    except FileExistsError as exc:
        print(exc)
        print("existing file will be appended instead...")
        df_card.to_csv(path_card, mode = 'a', header = False)
        df_bank.to_csv(path_bank, mode = 'a', header = False)
        #df_demo.to_csv(csv_path_demo, mode = 'a', header = False)
#return 'File preprocessed and CSVs saved in the working directory (C:\Users\Username\)'
#%%
    '''
            CONVERSION OF THE SPENDING REPORTS

    For testing purposes which does not include randomized IDs as part of the name and allows loading a constant name
    Test the functionality and reassert some columns after CSV is generated
    AFTER CSV IS GENERATED(in pred_func):
        Reassert datetime objects to all date columns
        Set Transaction date as index
    '''
    raw = os.getcwd()
    date_of_creation = dt.today().strftime('%m-%d-%Y_%Hh-%mmin')

    csv_path_msp = os.path.abspath(os.path.join(raw, date_of_creation + '_MONTHLY_REPORT_ALL_USERS' + '.csv'))
    csv_path_wsp = os.path.abspath(os.path.join(raw, date_of_creation + '_WEEKLY_REPORT_ALL_USERS' + '.csv'))
    csv_path_dsp = os.path.abspath(os.path.join(raw, date_of_creation + '_DAILY_REPORT_ALL_USERS' + '.csv'))

    try:
        spending_metrics_monthly.to_csv(csv_path_msp)
        spending_metrics_weekly.to_csv(csv_path_wsp)
        spending_metrics_daily.to_csv(csv_path_dsp)
    except FileExistsError as exc:
        print(exc)
        print("existing file will be appended instead...")
        spending_metrics_monthly.to_csv(csv_path_msp, mode = 'a', header = False)
        spending_metrics_weekly.to_csv(csv_path_wsp, mode = 'a', header = False)
        spending_metrics_daily.to_csv(csv_path_dsp, mode = 'a', header = False)
#close the function with return xx to avoid error 500 when querying the URL and have a message showing up instead
    return 'Preprocessing completed.'