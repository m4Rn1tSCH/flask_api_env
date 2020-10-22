#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:54:57 2020

@author: bill
"""
from ml_code.model_data.raw_data_connection import pull_df
import pandas as pd
import os
from datetime import datetime as dt

# pull vendor_list
vendors_df = pd.read_csv(os.path.abspath("/Users/bill/OneDrive - Envel/Bill_vendors_list.csv"))
vendors = vendors_df['MerchantName'].unique().tolist()

# unique values of all transactions of ydb
us_states1 = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA']
us_states2 = ['HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD']
us_states3 = ['MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ']
us_states4 = ['NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC']
us_states5 = ['SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
#1, 2, 3, 4, 5
yod_list = []
for i in us_states2:
    state_mer = pull_df(state=i,
                        spending_report=False,
                        plots=False)
    un_merchants = state_mer['primary_merchant_name'].unique().tolist()
    yod_list.append(un_merchants)
for j in yod_list:
    vendors.append(j)

#merch_df = pd.DataFrame(data = {'merchant_name':vendors})
merch_df = pd.DataFrame(data = {'merchant_name':un_merchants})
# export to csv
raw = os.getcwd()
date_of_creation = dt.today().strftime('%m-%d-%Y_%Hh-%mmin')

path = os.path.abspath(os.path.join(raw, date_of_creation + '_VENDORS_MERGED_card2' + '.csv'))

try:
    merch_df.to_csv(path)
    print(f"File in: {raw}")
except FileExistsError as exc:
    print(exc)
    print(f"existing file will be appended: {raw}")
    merch_df.to_csv(path, mode = 'a', header = False)
#%%
df1 = pd.read_csv(os.path.abspath("/Users/bill/OneDrive - Envel/09-15-2020_08h-09min_VENDORS_MERGED_card1.csv"))
df2 = pd.read_csv(os.path.abspath("/Users/bill/OneDrive - Envel/09-15-2020_09h-09min_VENDORS_MERGED_card2.csv"))
df3 = pd.read_csv(os.path.abspath("/Users/bill/OneDrive - Envel/09-15-2020_08h-09min_VENDORS_MERGED_card3.csv"))
df4 = pd.read_csv(os.path.abspath("/Users/bill/OneDrive - Envel/09-15-2020_09h-09min_VENDORS_MERGED_card4.csv"))
df5 = pd.read_csv(os.path.abspath("/Users/bill/OneDrive - Envel/09-15-2020_09h-09min_VENDORS_MERGED_card5.csv"))

y_df = pd.concat([df1, df2, df3, df4, df5],axis = 0)
y_df.drop_duplicates()
y_df = y_df.drop(columns = ['Unnamed: 0'])
##
raw = os.getcwd()
date_of_creation = dt.today().strftime('%m-%d-%Y_%Hh-%mmin')
path = os.path.abspath(os.path.join(raw, date_of_creation + '_UNIQUE_VENDORS' + '.csv'))
##
y_df.to_csv(path)
