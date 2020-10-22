# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:21:10 2020

@author: bill-
"""

'''
This module conducts calculations if fed with a dataframe
determines:
    daily spending (mean/std)
    weekly spending (mean/std)
    monthly spending (mean/std)
the report is saved as a pandas df and converted to a CSV
LOCAL:the CSV is saved in the current working directory of the device
AWS:the CSV is saved in ././injection/*.csv
4/9/2020: works fine for bank and card panel
'''
import os
from datetime import datetime as dt
import pandas as pd

#in flask body with variable input
#allows to input file
#self in python: self is updating an instance variable of its own function
#in this case the instance is the dataframe fed to the method and that is supposed to be processed

def spending_report(df):
    #%%
    '''
    Datetime engineering for card and bank panel
    These columns help with reporting like weekly or monthly expenses and
    improve prediction of re-occurring transactions
    '''
    for col in list(df):
        if df[col].dtype == 'datetime64[ns]':
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_week"] = df[col].dt.week
            df[f"{col}_weekday"] = df[col].dt.weekday
    df.reset_index(drop = True, inplace = True)
    '''
    Addition of feature columns for additive spending on a weekly; monthly; daily basis
    These dataframes are then convertable to a CSV for reporting purposes or could be shown in the app
    As of 4/2/2020 the spending report generates a file-wide dataframe based on all users
    IN CARD PANEL: debit will decrease its balance and credit will increase it (liability on the bank's part)
    IN BANK PANEL: debit will decrease its balance and credit will increase it (liability on the bank's part)
    '''
    #total throughput of money
    total_throughput = df['amount'].sum()
    #monthly figures
    net_monthly_throughput = df['amount'].groupby(df['optimized_transaction_date_month']).sum()
    avg_monthly_throughput = df['amount'].groupby(df['optimized_transaction_date_month']).apply(lambda x: x.mean())
    #CHECK VIABILITY OF SUCH VARIABLES
    monthly_gain = df['amount'][df['transaction_base_type'] == 'credit'].groupby(df['optimized_transaction_date_month']).sum()
    monthly_expenses = df['amount'][df['transaction_base_type'] == 'debit'].groupby(df['optimized_transaction_date_month']).sum()
    #weekly figures
    net_weekly_throughput = df['amount'].groupby(df['optimized_transaction_date_week']).sum()
    avg_weekly_throughput = df['amount'].groupby(df['optimized_transaction_date_week']).apply(lambda x: x.mean())
    #CHECK VIABILITY OF SUCH VARIABLES
    weekly_gain = df['amount'][df['transaction_base_type'] == "credit"].groupby(df['optimized_transaction_date_week']).sum()
    weekly_expenses = df['amount'][df['transaction_base_type'] == "debit"].groupby(df['optimized_transaction_date_week']).sum()
    #daily figures
    net_daily_spending = df['amount'].groupby(df['optimized_transaction_date_weekday']).sum()
    avg_daily_spending = df['amount'].groupby(df['optimized_transaction_date_weekday']).apply(lambda x: x.mean())
    #CHECK VIABILITY OF SUCH VARIABLES
    daily_gain = df['amount'][df['transaction_base_type'] == "credit"].groupby(df['optimized_transaction_date_weekday']).sum()
    daily_expenses = df['amount'][df['transaction_base_type'] == "debit"].groupby(df['optimized_transaction_date_weekday']).sum()

    #report for users about their spending patterns, given in various intervals
    try:
        print(f"The total turnover on the account has been ${total_throughput}")
        #print("................................................................")
        spending_metrics_monthly = pd.DataFrame(data = {'Average Monthly Spending':avg_monthly_throughput,
                                                        'Monthly Turnover':net_monthly_throughput,
                                                        'Monthly Inflow':monthly_gain,
                                                        'Monthly Outflow':monthly_expenses})
        #print(spending_metrics_monthly)
        #print(".................................................................")
        spending_metrics_weekly = pd.DataFrame(data = {'Average Weekly Spending':avg_weekly_throughput,
                                                       'Weekly Turnover':net_weekly_throughput,
                                                       'Weekly Inflow':weekly_gain,
                                                       'Weekly Outflow':weekly_expenses})
        #print(spending_metrics_weekly)
        #print(".................................................................")
        spending_metrics_daily = pd.DataFrame(data = {'Average Daily Spending':avg_daily_spending,
                                                      'Daily Turnover':net_daily_spending,
                                                      'Daily Inflow':daily_gain,
                                                      'Daily Outlow':daily_expenses})
        #print(spending_metrics_daily)
    except:
        print("You do not have enough transactions yet. But we are getting there...")

    '''
            CONVERSION OF THE SPENDING REPORTS - ALL USERS
    For testing purposes which does not include randomized IDs as part of the name and allows loading a constant name
    calculations are incorporating all users simultaneously!
    '''
    #local working directory
    raw = os.getcwd()
    #folder when executed on the AWS instance
    #aws = os.mkdir('/injection')
    date_of_creation = dt.today().strftime('%m-%d-%Y_%Hh-%mmin')

    csv_path_msp = os.path.abspath(os.path.join(raw, date_of_creation + '_MONTHLY_REPORT' + '.csv'))
    csv_path_wsp = os.path.abspath(os.path.join(raw, date_of_creation + '_WEEKLY_REPORT' + '.csv'))
    csv_path_dsp = os.path.abspath(os.path.join(raw, date_of_creation + '_DAILY_REPORT' + '.csv'))

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
    return 'Spending report generated; CSV-file in current working directory.'

#add this part at the end to make the module executable as script
#takes arguments here (df)
#
    if __name__ == "__main__":
        import sys
        spending_report(int(sys.argv[1]))