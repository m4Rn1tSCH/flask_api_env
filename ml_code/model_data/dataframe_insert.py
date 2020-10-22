#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:08:51 2020

@author: bill
"""

from ml_code.model_data.SQL_connection import create_connection
from ml_code.model_data.raw_data_connection import pull_df
import psycopg2
from psycopg2 import OperationalError

#TEST_DF
# df = pull_df(rng=22,
#              spending_report=False,
#              plots=False)
#########################

def df_insert_query(df):
    
    """
    Parameters
    ---------
    df. pandas dataframe that is to be inserted into a databank.
    Returns
    --------
    'insert complete' msg.
    """



    # create the placeholders for the columns that will be fitted with values
    #col_records = ", ".join(["%s"] * len(df.columns))

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
                          test_col_9, test_col_10, test_col_11)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
                                               i.optimized_transaction_date]))

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

    return'insert completed'
