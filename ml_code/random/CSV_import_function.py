'''
CONVERSION from .csv tp pandas_df
'''

import pandas as pd

def csv_import(path):

    df = pd.read_csv(path)
    print('Successfully imported csv file into pandas df')

    return df

# /Users/michael/dev/envel/envel-machine-learning/data/yodlee_data
# section_10_yodlee.csv section_1_yodlee.csv  section_2_yodlee.csv  section_3_yodlee.csv  section_4_yodlee.csv  section_5_yodlee.csv  section_6_yodlee.csv  section_7_yodlee.csv  section_8_yodlee.csv  section_9_yodlee.csv
