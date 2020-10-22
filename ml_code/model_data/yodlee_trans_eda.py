from psycopg2 import OperationalError
import numpy as np
import pandas as pd
from datetime import datetime as dt

# FILE IMPORTS FOR NOTEBOOKS
from SQL_connection import execute_read_query, create_connection
import PostgreSQL_credentials as acc

def yodlee_sectional_df(section=1):
    connection = create_connection(db_name=acc.YDB_name,
                                    db_user=acc.YDB_user,
                                    db_password=acc.YDB_password,
                                    db_host=acc.YDB_host,
                                    db_port=acc.YDB_port)

    data = []
    fields = ['unique_mem_id', 'amount', 'transaction_base_type',
              'transaction_category_name', 'optimized_transaction_date']

    # number of days in month for 2019 (yodlee db year)
    days_monthly = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    transaction_month = [x for x in range(1, 13)]
    # transaction_categorical_name to include as bills
    bills_categories = ['Cable/Satellite/Telecom', 'Healthcare/Medical', 'Insurance',
                        'Mortgage', 'Rent', 'Subscriptions/Renewals', 'Utilities',
                        'Loans', 'Education']

    try:
        filter_query = f"(select {', '.join(field for field in fields)} from card_record where unique_mem_id in (select unique_mem_id from user_demographic order by unique_mem_id asc limit 10000 offset {10000*(section-1)})) union all (select {', '.join(field for field in fields)} from bank_record where unique_mem_id in (select unique_mem_id from user_demographic order by unique_mem_id asc limit 10000 offset {10000*(section-1)}))"
        transaction_query = execute_read_query(connection, filter_query)
        main_df = pd.DataFrame(transaction_query, columns=fields)
        print(f"{len(transaction_query)} transactions.")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        connection.rollback

    for num, user in enumerate(main_df.groupby('unique_mem_id')):
        print(f"user {user[0]}, {num+1}/10000 users, {round(((num+1)/10000)*100, 2)}%.")

        df = pd.DataFrame(user[1], columns=fields)
        df['amount'] = df['amount'].astype('float64')
        # add date columns
        df['optimized_transaction_date'] = pd.to_datetime(
            df['optimized_transaction_date'])
        df["transaction_month"] = df['optimized_transaction_date'].dt.month
        df["transaction_day"] = df['optimized_transaction_date'].dt.day

        monthly_income = df['amount'][df['transaction_base_type'].eq('credit')].groupby(
            df['transaction_month']).sum().round(2)
        # fill all 12 months as some users missing data
        for i in transaction_month:
            try:
                monthly_income[i]
            except:
                monthly_income[i] = 0
        monthly_expenses = df['amount'][df['transaction_base_type'].eq('debit')].groupby(
            df['transaction_month']).sum().round(2)
        monthly_savings = round(monthly_income - monthly_expenses, 2)

        # total bills per month according to categories, then remove bills transactions from df
        monthly_bills = df['amount'][df['transaction_category_name'].isin(bills_categories)].groupby(
            df['transaction_month']).sum().round(2)
        for i in transaction_month:
            try:
                monthly_bills[i]
            except:
                monthly_bills[i] = 0
        df = df[~df['transaction_category_name'].isin(bills_categories)]

        # ai devisions
        monthly_emergency = round(monthly_income * 0.1, 2)
        monthly_vault = round(monthly_income * 0.1, 2)
        monthly_cash = round(
            monthly_income - (monthly_bills * 1.1) - monthly_emergency - monthly_vault, 2)
        monthly_daily = round(monthly_cash / days_monthly, 2)

        # calculations if ai mode was on
        daily_overspent = []
        daily_underspent = []
        for month, days in enumerate(days_monthly, start=1):
            daily_cash = monthly_daily[month]
            daily_total_underspent = 0
            daily_total_overspent = 0
            daily_expenses = df['amount'][df['transaction_base_type'].eq('debit') & df['transaction_month'].eq(month)].groupby(
                df['transaction_day']).sum()
            for day in range(1, days + 1):
                try:
                    daily_expense = daily_expenses[day]
                except:
                    daily_expense = 0
                daily_left = round(daily_cash - daily_expense, 2)
                if daily_left > 0:
                    daily_total_underspent = daily_total_underspent + daily_left
                else:
                    daily_total_overspent = daily_total_overspent - daily_left
            daily_overspent.append(round(daily_total_overspent, 2))
            daily_underspent.append(round(daily_total_underspent, 2))

        # calculations for nett benefit from ai
        monthly_vault_end = monthly_vault - daily_overspent
        monthly_emergency_end = monthly_emergency + daily_underspent
        monthly_savings_envel = monthly_emergency_end + abs(monthly_vault_end)

        for i in transaction_month:
            try:
                data.append([user[0], monthly_income[i], monthly_savings[i], monthly_savings_envel[i]])
            except:
              print(f"missing values for month {i} of user {user[0]}")

    columns = ['unique_mem_id', 'monthly_income', 'monthly_savings', 'monthly_savings_envel']
    trans_df = pd.DataFrame(data, columns=columns)

    return trans_df


# a function to run through when already have the transactions df that get from previous function
def trans_calculations(df):
    results = {}
    df_grouped = df.groupby('unique_mem_id')
    bracket_list = [[0, 15], [15, 30], [30, 75], [75, 100], [100, 200], [200, 300], [300, 500], [500, 99999999]]

    for bracket in bracket_list:
        trans_df_grouped = df_grouped.filter(lambda x: (bracket[0]*1000) <= x['monthly_income'].sum() < (bracket[1]*1000)).groupby('unique_mem_id')

        total_users_bracket = len(trans_df_grouped.size())
        avg_savings_pu = np.array(trans_df_grouped['monthly_savings'].mean())
        aggr_savings_pu = np.array(trans_df_grouped['monthly_savings'].sum())
        avg_savings_pupm = avg_savings_pu.mean().round(2)

        avg_savings_pu_envel = np.array(trans_df_grouped['monthly_savings_envel'].mean())
        aggr_savings_pu_envel = np.array(trans_df_grouped['monthly_savings_envel'].sum())
        avg_savings_pupm_envel = avg_savings_pu_envel.mean().round(2)

        aggr_savings = aggr_savings_pu.sum().round(2)
        aggr_savings_envel = aggr_savings_pu_envel.sum().round(2)

        avr_inc_pu = np.array(trans_df_grouped['monthly_income'].mean())
        avr_inc_pu_tot = avr_inc_pu.mean().round(2)
        num_living_ptp = (avg_savings_pu < avr_inc_pu).sum()
        num_living_ptp_envel = (avg_savings_pu_envel < avr_inc_pu).sum()

        num_400_problem = (aggr_savings_pu < 4800).sum()
        num_400_problem_envel = (aggr_savings_pu_envel < 4800).sum()

        results[f'bracket_{bracket[0]}_{bracket[1]}'] = {'total_users_bracket': total_users_bracket,
                                                        'avr_income_pu': avr_inc_pu_tot,
                                                        'avg_savings_pupm': avg_savings_pupm,
                                                        'avg_savings_pupm_envel': avg_savings_pupm_envel,
                                                        'aggr_savings': aggr_savings,
                                                        'aggr_savings_envel': aggr_savings_envel,
                                                        'num_living_ptp': num_living_ptp,
                                                        'num_living_ptp_envel': num_living_ptp_envel,
                                                        'num_400_problem': num_400_problem,
                                                        'num_400_problem_envel': num_400_problem_envel}

    return results

# a function to import the yodlee trans data from csv and run through the transactions function above
def yodlee_full(path):
    full_df = pd.read_csv(f'{path}/full_yodlee.csv')

    results = trans_calculations(full_df)

    return results
