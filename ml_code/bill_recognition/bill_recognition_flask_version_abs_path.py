##input_1: csv with transactions
##input_2: xlsx with transactions
##output: list with detected bills

###FLASK FUNCTION###
from flask import Flask
####PACKAGES OF THE MODULE######
import pandas as pd
import os
import re
##from sqlalchemy import create_engine
############################################
###SQL-CONNECTION TO QUERY THE VENDOR FILE
###Create engine named engine
##engine = create_engine('sqlite:///Chinook.sqlite')

##Open engine connection: con
##con = engine.connect()

##Perform query: rs
##rs = con.execute("SELECT * from <<DB_FOLDER>>")

#Save results of the query to DataFrame: df
##df = pd.DataFrame(rs.fetchall())

##Close connection
##con.close()
##############################################
#%%
######LOADING THE TWO FILES#####

transaction_file = r"C:\Users\bill-\Desktop\TransactionsD_test.csv"
path_1 = transaction_file.replace(os.sep,'/')
transactions = ''.join(('', path_1, ''))
#%%
vendor_file = r"C:\Users\bill-\Dropbox\Nan\Archived\BillVendors_Only.xlsx"
path_11 = vendor_file.replace(os.sep,'/')
vendors = ''.join(('', path_11, ''))
#%%
#exclude these merchants as repetitive payments
blacklist = ['Uber', 'Lyft', 'Paypal', 'E-ZPass']
bills_found = []
#%%
#############SETTING UP THE APP##########

app = Flask(__name__)
####TRIGGER URL#####

@app.route('/')

#GIVE THE FUNCTION A NAME TO GENERATE A URL IT IS BEING CALLED WITH
#TRANSACTION_INPUT: FILE WITH CUSTOMER TRANSACTIONS
#VENDORS: LIST OF VENDORS THAT OFFER SUBSCRIPTION SERVICES AND RECURRING BILLING
#EXCLUDE:LIST ITEMS TO EXLCUDE FROM APPENDAGE TO A FOLE_LIST
def bill_recognition(transaction_input = transactions,
                     vendor_list = vendors,
                     exclude = blacklist):
    ##keep this for tests on other OSs and to avoid paths problems
    os.getcwd()
    ######FUNCTION OF THE MODULE########
    #load csv
    df = pd.read_csv(transaction_input, header = 0, names = ['date',
                                                             'category',
                                                             'trans_cat',
                                                             'subcat',
                                                             'shopname',
                                                             'amount'])
    df.head(n = 3)
    len(df.index)
    #if tokenizing error arises; might be due to pandas generated columns names with an \r
    #then the discrepancy causes an error; specify separator explicitly to fix
    df1 = pd.read_excel(vendor_list, header = 0, names = ['MerchantName',\
                                                              'BillCategory'])
    print("loading the vendor list...")
    BillVendors_uniqueVals = df1['MerchantName'].unique()
    BillVendors = BillVendors_uniqueVals.tolist()
    ################
    bills_found = []
    #statements = list of bank statement strings
    for i in range(len(df.index)):
        descriptions = str(df.iloc[i]['shopname']).lower()
        for BillVendor in BillVendors:
            #BillVendor = BillVendor.lower()
            #re.I makes the process ignore lower/upper case
            if re.search(BillVendor, descriptions, flags = re.I):
                # append to bill_found list
                bills_found.append(descriptions)
                print("bill found")
        else:
            print("no known bill found :(")
    #recurring bills have breen written to a list

#iterate through the elements of bills_found
list_of_int = []
#convert the bills_found list to a tuple that is hashable
#bills_found is a key of a dictionary and is hashable
bill_dict = {tuple(list_of_int): bills_found}
for i in range(len(bills_found)):
    if re.search(exclude, bills_found, flags = re.I):
        # remove from bill_found list
        bills_found.remove(descriptions)
        print("blacklisted bill removed")
    else:
            pass
#recurring bills have breen written to a list

############################################
###SQL-CONNECTION TO QUERY THE VENDOR FILE
###Create engine named engine
##engine = create_engine('sqlite:///Chinook.sqlite')

##Open engine connection: con
##con = engine.connect()

##Perform query: rs
##rs = con.execute("SELECT * from <<DB_FOLDER>>")

#Save results of the query to DataFrame: df
##df = pd.DataFrame(rs.fetchall())

##Close connection
##con.close()
##############################################

############################################
###SQL-CONNECTION TO QUERY THE VENDOR FILE
###Create engine named engine
##engine = create_engine('sqlite:///Chinook.sqlite')

##Open engine connection: con
##con = engine.connect()

##Perform query: rs
##rs = con.execute("UPDATE * from <<DB_FOLDER>>")
        ##UPDATE <DB> SET <parameter> WHERE users.name = ?

#Save results of the query to DataFrame: df
##df = pd.DataFrame(rs.fetchall())

##Close connection
##con.close()
##############################################