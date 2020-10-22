# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:51:40 2020

@author: bill-
"""

'''
This script contains all SQL components
-make a connection to the Yodlee DB
-insert records to it
-delete records

when it throws error about transaction blocked enter: rollback
this reverts old incorrect queries

THE SCRIPT ONLY CONDUCTS QUERIES; THE OUTPUT IS A TUPLE!
THE MODULE IT IS CONNECTED TO WILL CONVERT THE PULLED CONTENT TO A DF
'''

#establish a connection to the Yodlee DB
import psycopg2
from psycopg2 import OperationalError
from psycopg2 import pool
#import PostgreSQL_access
#%%
name = "postgres"
user = "envel_yodlee"
pw = "Bl0w@F1sh321"
host = "envel-yodlee-datasource.c11nj3dc7pn5.us-east-2.rds.amazonaws.com"
port = "5432"
#create_connection(name, user, pw, host, port)
#%%
#assign connection object as variable + use in further functions
def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print(f"Connection to PostgreSQL {db_name} successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection
#%%
def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except OperationalError as e:
        print(f"The error '{e}' occurred")
#%%
#example query for transaction in MA
#select_users = "SELECT * FROM bank_record WHERE state = 'MA'"
#generates a tuple output
#transaction_query = execute_read_query(connection, select_users)
#%%
    #insert a value into the DB
def insert_val(query_string):

    '''
    sql_query: PostgreSQL query command. Engulf query in triple letter strings
                """example query here"""
    returns
    ------
    edit_msg
    '''
# SQL query example
    # create_users = """
    # INSERT INTO
    #   users (name, age, gender, nationality)
    # VALUES
    #   ('James', 25, 'male', 'USA'),
    #   ('Leila', 32, 'female', 'France'),
    #   ('Brigitte', 35, 'female', 'England'),
    #   ('Mike', 40, 'male', 'Denmark'),
    #   ('Elizabeth', 21, 'female', 'Canada');
    # """

    sql_query = query_string
    execute_query(connection, sql_query)
    return 'edit_msg'



# alternative version

def insert_val_alt(table, columns, insertion_val):

    '''
    table: string. Table in the databank to be amended.
    columns: list. no letterstrings to pass the columns that are to be amended
    insertion_val: tuple. Pass information inside a tuple with value in letter strings
                    (only for PostgreSQL); separated by commas
    returns
    ------
    edit_msg
    '''
# Tuple example
    # tuples = [
    #     ("James", 25, "male", "USA"),
    #     ("Leila", 32, "female", "France"),
    #     ("Brigitte", 35, "female", "England"),
    #     ("Mike", 40, "male", "Denmark"),
    #     ("Elizabeth", 21, "female", "Canada"),
    #     ]

    tuple_values = insertion_val

    # create the placeholders for the columns that will be fitted with values
    tuple_records = ", ".join(["%s"] * len(tuple_values))
    insert_query = (
        f"INSERT INTO {table} ({columns}) VALUES {tuple_records};"
        )

    connection.autocommit = True
    cursor = connection.cursor()
    cursor.execute(insert_query, tuple_values)
    return 'edit_msg'
#%%
def delete_val():
    #delete comments
    delete_comment = "DELETE FROM comments WHERE id = 2"
    execute_query(connection, delete_comment)
    return 'values deleted'
#%%
'''
IMPORTANT: This closes all connections even those that are in use by applications!
    Use with caution!
'''
#close a single connection pool
def close_connection():
    pool.SimpleConnectionPool.closeall


    if __name__ == "__main__":
        import sys
        close_connection(int(sys.argv[1]))
