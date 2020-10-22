import pandas as pd
import os
import re
##keep this for tests on other OSs and to avoid paths problems
os.getcwd()
#%%
#Pay attention if it is a CSV or Excel file to avoid tokenization errors and separator errors
#link for the transaction csv
transaction_input = r"C:\Users\bill-\Desktop\TransactionsD_test.csv"
path_1 = transaction_input.replace(os.sep,'/')
transaction_input = ''.join(('', path_1, ''))
#%%
#
vendor_list_input = r"C:\Users\bill-\Dropbox\Nan\Archived\BillVendors_Only.xlsx"
path_11 = vendor_list_input.replace(os.sep,'/')
vendor_list_input = ''.join(('', path_11, ''))
#%%
#load csv
df = pd.read_csv(transaction_input, header = 0, names = ['date',
                                                         'category',
                                                         'trans_cat',
                                                         'subcat',
                                                         'shopname',
                                                         'amount'])
df.head(n = 3)
len(df.index)
#%%
# figure out repetitive payments
# exclude these merchants as repetitive payments
blacklist = ['Uber', 'Lyft', 'Paypal', 'E-ZPass']
#%%
#if tokenizing error arises; might be due to pandas generated columns names with an \r
#then the discrepancy causes an error; specify separator explicitly to fix
df1 = pd.read_excel(vendor_list_input, header = 0, names = ['MerchantName',\
                                                          'BillCategory'])
print("loading the vendor list...")
BillVendors_uniqueVals = df1['MerchantName'].unique()
BillVendors = BillVendors_uniqueVals.tolist()

#change the elements to lower case only
#for BillVendor in BillVendors:
#
bills_found = []
#%%
#statements = list of bank statement strings
for i in range(len(df.index)):
    descriptions = str(df.iloc[i]['shopname']).lower()
    #descriptions = descriptions.lower()
    #print(descriptions)
    #print(BillVendor)
    for BillVendor in BillVendors:
        #BillVendor = BillVendor.lower()
        ###first if-loop
        #re.I makes the process ignore lower/upper case
        if re.search(BillVendor, descriptions, flags = re.I):
            # append to bill_found list
            bills_found.append(descriptions)
            print("bill found")
        ###second if-loop
    else:
        print("no known bill found :(")
#%%
#iterate through the elements of bills_found
for i in range(len(bills_found)):
    if re.search(blacklist, bills_found, flags = re.I):
        # remove from bill_found list
        bills_found.remove(descriptions)
        print("blacklisted bill removed")
    else:
            pass
#recurring bills have breen written to a list
