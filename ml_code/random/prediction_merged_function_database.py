'''
HIDDEN BLOCKER: COLUMNS CITY CONTAINS "NAN" AS NON-PANDAS NAN AND
SUBSEQUENTLY DOES NOT SHOWN UP AS NULL-VALUE OR IS FIXED BY .FILLNA()
REQUIRES .REPLACE TO GET RID OF IT
'''
import pandas as pd
import os
import matplotlib.pyplot as plt

import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
#from datetime import datetime
#import seaborn as sns
#plt.rcParams["figure.dpi"] = 600
#plt.rcParams['figure.figsize'] = [12, 10]
#%%
'''
Setup of the function to merge every single operation into one function that is then called by the flask connection/or SQL
contains: preprocessing, splitting, training and eventually prediction
'''
#CONNECTION TO FLASK/SQL
#INSERT FLASK CONNECTION SCRIPT HERE
###########################################
#loading the simplified applications
#from flask import Flask
#app = Flask(__Preprocessor__)

##put address here
#@app.route('/')
#def hello_world():
#    return 'Hello, World!'
#route tells what URL should trigger the function
#use __main__ only for the actual core of the application
# pick unique names for particular functions if these are imported later
#DONT CALL APPLICATIONS FLASK.PY TO AVOID CONFLICTS WITH FLASK

#RUN THE APPLICATION
#flask command or -m swith in Python

########SETTING THE ENVIRONMENT VARIABLE#######
#$ export FLASK_APP = C:\Users\bill-\OneDrive\Dokumente\Docs Bill\TA_files\functions_scripts_storage\Python_prediction_merged_function_database.py
#$ flask run
# * Running on http://127.0.0.1:5000/

####COMMAND PROMPT#####
#C:\path\to\app>set FLASK_APP=hello.py

####for production use##
#$ flask run --host=0.0.0.0
############################################


#INSERT SQL CONNECTION HERE
############################################
###SQL-CONNECTION TO QUERY THE VENDOR FILE
###Create engine
##engine = create_engine('sqlite:///Chinook.sqlite')

##Open engine connection: con
##con = engine.connect()

##Perform query: rs
##rs = con.execute("SELECT * from <<DB_FOLDER>>")

#Save results df
##df = pd.DataFrame(rs.fetchall())

##Close connection
##con.close()
##############################################
#%%
######LOADING THE TRANSACTION FILE#####
#transaction_file = r"C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\2020-01-28 envel.ai Working Class Sample.xlsx"
#path_1 = transaction_file.replace(os.sep,'/')
#transactions = ''.join(('', path_1, ''))
'''
The preprocessed files are CSV; the function will load all CSVs and pick label and features
for testing purposes:
'''
#relative_t_path = './*.csv'
#Windows paths fo files
########################
#"C:\Users\bill-\Desktop\03-04-2020_CARD_PANEL.csv"
#"C:\Users\bill-\Desktop\03-04-2020_BANK_PANEL.csv"
#"C:\Users\bill-\Desktop\03-04-2020_DEMO_PANEL.csv"
#######################
#Mac paths to files
#######################
#/Users/bill/OneDrive - Envel/03-04-2020_BANK_PANEL.csv
#/Users/bill/OneDrive - Envel/03-04-2020_CARD_PANEL.csv
#/Users/bill/OneDrive - Envel/03-04-2020_DEMO_PANEL.csv
#####################
mac_walk_path = '/Users/bill/Desktop/'
windows_walk_path = 'C:/Users/bill-/Desktop/'
#%%
#import files and append all directory paths to a list
#basepath = 'C:/Users/bill-/Desktop/Harvard_Resumes'
#path_list = []
#Walking a directory tree and printing the names of the directories and files
#for dirpath, dirnames, filename in os.walk(basepath):
#    print(f'Found directory: {dirpath}')
#    for file in filename:
#        if os.path.isfile(file):
#            print("file found and appended")
#        path_list.append(os.path.abspath(os.path.join(dirpath, file)))
#%%

#Write the pattern as a folder or file pattern
path_abs = os.path.abspath(os.path.join(windows_walk_path))
pattern = '*.csv'
directory = os.path.join(path_abs, pattern)
#Save all file matches: csv_files
csv_files = glob.glob(directory)
#Print the file names
#list has weirdly ranked names!
#ON MACBOOK;demo occurs before bank panel; needs to be taken care of in tests
print(csv_files)
#%%
def predict_needed_value(preprocessed_input):
#order different depending on OS!
#use parser in list to attach right csv to right variable
    df_card_rdy = pd.read_csv(csv_files[4])
    df_bank_rdy = pd.read_csv(csv_files[3])
    df_demo_rdy = pd.read_csv(csv_files[5])
    #the conversion to csv has removed the old index and left date columns as objects
    #conversion is needed to datetime objects
    #usage of optimized transaction date is recommended


    df_card_rdy.set_index("optimized_transaction_date", drop = False, inplace = False)
#columns transaction date is available here and can be normally used;
    df_bank_rdy.set_index("optimized_transaction_date", drop = False, inplace = False)
    try:
        y_cp_city.replace("nan", "unknown")
        y_cp_city.fillna(value = 'unknown')
    except:
        pass

#%%
#3/24/2020 : preprocessing works completely; might not be necessary anymore
#remove column transaction_date temporarily;
#    date_card_col = ['transaction_date', 'post_date', 'file_created_date',
#                'optimized_transaction_date', 'swipe_date', 'panel_file_created_date']
#    date_bank_col = ['transaction_date', 'post_date', 'file_created_date',
#                'optimized_transaction_date', 'swipe_date', 'panel_file_created_date']
#    for elements in list(date_card_col):
#        df_card_rdy[elements] = pd.to_datetime(df_card_rdy[elements])
#    for elements in list(date_bank_col):
#        df_bank_rdy[elements] = pd.to_datetime(df_bank_rdy[elements])
    #%%
    '''
    Columns preprocessed card_panel
    ['transaction_date', 'unique_mem_id', 'unique_card_account_id',
       'unique_card_transaction_id', 'amount', 'currency', 'description',
       'transaction_date.1', 'post_date', 'transaction_base_type',
       'transaction_category_name', 'primary_merchant_name',
       'secondary_merchant_name', 'city', 'state', 'zip_code',
       'transaction_origin', 'factual_category', 'factual_id',
       'file_created_date', 'optimized_transaction_date',
       'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
       'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
       'account_type', 'account_source_type', 'account_score', 'user_score',
       'lag', 'is_duplicate', 'transaction_date_month',
       'transaction_date_week', 'transaction_date_weekday', 'post_date_month',
       'post_date_week', 'post_date_weekday', 'file_created_date_month',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']

    Columns preprocesed bank_panel
    ['transaction_date', 'unique_mem_id', 'unique_card_account_id',
       'unique_card_transaction_id', 'amount', 'currency', 'description',
       'transaction_date.1', 'post_date', 'transaction_base_type',
       'transaction_category_name', 'primary_merchant_name',
       'secondary_merchant_name', 'city', 'state', 'zip_code',
       'transaction_origin', 'factual_category', 'factual_id',
       'file_created_date', 'optimized_transaction_date',
       'yodlee_transaction_status', 'mcc_raw', 'mcc_inferred', 'swipe_date',
       'panel_file_created_date', 'update_type', 'is_outlier', 'change_source',
       'account_type', 'account_source_type', 'account_score', 'user_score',
       'lag', 'is_duplicate', 'transaction_date_month',
       'transaction_date_week', 'transaction_date_weekday', 'post_date_month',
       'post_date_week', 'post_date_weekday', 'file_created_date_month',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']
    '''
    #%%
'''
------------------------------------------------------------------------------
Prediction for transaction_category_name and RFE for significant features
'''
    ##SELECTION OF FEATURES AND LABELS FOR CARD PANEL
    #first prediction loop and stop
#    for col in df_card_rdy.columns:
#        if len(df_card_rdy[col]) != len(df_card_rdy.index):
#            df_card_rdy.drop(col)
#            print(f"{col} has been removed")
    y_cp = df_card_rdy['transaction_category_name']
    X_cp = df_card_rdy[['amount', #'city', 'state', 'zip_code',
        'post_date_month', 'post_date_week', 'post_date_weekday',
        'file_created_date_month',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]
    try:
        y_cp.replace("nan", "unknown")
        y_cp.fillna(value = 'unknown')
    except:
        pass
#%%
    ##SELECTION OF FEATURES AND LABELS FOR BANK PANEL
    #drop transaction and lagging columns as it disappears in card panel CSV
    #first prediction loop and stop
    #ALTERNATIVE
    #df[['col_1', 'col_2', 'col_3', 'col_4']]
#    for col in df_bank_rdy.columns:
#        if len(df_bank_rdy[col]) != len(df_bank_rdy.index):
#            df_bank_rdy.drop(col)
#            print(f"{col} has been removed")
    y_bp = df_bank_rdy['transaction_category_name']
    X_bp = df_bank_rdy[['amount', #'city', 'state', 'zip_code',
       'post_date_month', 'post_date_week',
       'post_date_weekday', 'file_created_date_month',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]
    try:
        y_bp.replace("nan", "unknown")
        y_cp.fillna(value = 'unknown')
    except:
        pass
    #%%
    #PICK FEATURES AND LABELS
    #columns = list(df_card)
    #print("enter the label that is to be predicted...; all other columns will remain and picked later as features ranked by prediction importance")
    #label_str = input("What should be predicted?")
    #X = columns.pop(columns.index(label_str))

    #X = list(df_card)
    #set the label
    #y = list(df_card).pop(list(df_card)('amount'))
    #%%
    #standard scaler takes the entire column list and also converted to a df with double square brackets
    #first fit/learn with training data
    #then transform with that object: train + test data
    scaler = StandardScaler()
    #fit_transform also separately callable; but this one is more time-efficient
    X_cp_scl = scaler.fit_transform(X_cp)
    X_bp_scl = scaler.fit_transform(X_bp)
    #%%
    #Kmeans clusters to categorize budget groups WITH SCALED DATA
    #5 different daily limit groups
    #determine number of groups needed or desired for
    kmeans = KMeans(n_clusters = 5, random_state = 10)
    cp_sclaed_clusters = kmeans.fit(X_cp_scl)
    kmeans = KMeans(n_clusters = 5, random_state = 10)
    bp_scaled_clusters = kmeans.fit(X_bp_scl)
    #%%
    #5 different daily limit groups
    #determine number of groups needed or desired for
    kmeans = KMeans(n_clusters = 5, random_state = 10)
    cp_clusters = kmeans.fit(X_cp)
    kmeans = KMeans(n_clusters = 5, random_state = 10)
    bp_clusters = kmeans.fit(X_bp)
    #%%
    #split into principal components for card panel
    pca = PCA(n_components = 2)
    cp_components = pca.fit_transform(X_cp_scl)
    #split into principla components for card panel
    bp_components = pca.fit_transform(X_bp_scl)
    #%%
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (15, 10), dpi = 800)
    #styles for title: normal; italic; oblique
    ax[0].scatter(cp_components[:, 0], cp_components[:, 1], c = cp_clusters.labels_)
    ax[0].set_title('Plotted Principal Components of CARD PANEL', style = 'oblique')
    ax[0].legend(cp_clusters.labels_)
    ax[1].scatter(bp_components[:, 0], bp_components[:, 1], c = bp_clusters.labels_)
    ax[1].set_title('Plotted Principal Components of BANK PANEL', style = 'oblique')
    ax[1].legend(bp_clusters.labels_)
    #principal components of bank panel has better results than card panel with clearer borders
    #%%
    #TRAIN TEST SPLIT FOR CARD PANEL
    #Train Size: percentage of the data set
    #Test Size: remaining percentage
    #from sklearn.model_selection import train_test_split

    X_cp_train, X_cp_test, y_cp_train, y_cp_test = train_test_split(X_cp, y_cp, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("CARD PANEL-TRANSACTION_CATEGORY_NAME")
    print(f"Shape of the split training data set: X_cp_train:{X_cp_train.shape}")
    print(f"Shape of the split training data set: X_cp_test: {X_cp_test.shape}")
    print(f"Shape of the split training data set: y_cp_train: {y_cp_train.shape}")
    print(f"Shape of the split training data set: y_cp_test: {y_cp_test.shape}")
    #%%
    #TRAIN TEST SPLIT FOR BANK PANEL
    #Train Size: percentage of the data set
    #Test Size: remaining percentage
    #from sklearn.model_selection import train_test_split

    X_bp_train, X_bp_test, y_bp_train, y_bp_test = train_test_split(X_bp, y_bp, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("BANK PANEL-TRANSACTION_CATEGORY_NAME")
    print(f"Shape of the split training data set: X_bp_train:{X_bp_train.shape}")
    print(f"Shape of the split training data set: X_bp_test: {X_bp_test.shape}")
    print(f"Shape of the split training data set: y_bp_train: {y_bp_train.shape}")
    print(f"Shape of the split training data set: y_bp_test: {y_bp_test.shape}")
    #%%
    #PASS TO RECURSIVE FEATURE EXTRACTION CARD PANEL
    #build a logistic regression and use recursive feature elimination to exclude trivial features
    log_reg = LogisticRegression(C = 0.01, class_weight = None, dual = False,
                                   fit_intercept = True, intercept_scaling = 1,
                                   l1_ratio = None, max_iter = 1000,
                                   multi_class = 'auto', n_jobs = None,
                                   penalty = 'l2', random_state = None,
                                   solver = 'lbfgs', tol = 0.0001, verbose = 0,
                                   warm_start = False)
    #create the RFE model and select the eight most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_cp_train, y_cp_train)
    #selected attributes
    print('Selected features: %s' % list(X_cp_train.columns[rfe.support_]))
    print(rfe.ranking_)

    #Use the Cross-Validation function of the RFE module
    #accuracy describes the number of correct classifications
    rfecv = RFECV(estimator = log_reg, step = 1, cv = 8, scoring = 'accuracy')
    rfecv.fit(X_cp_train, y_cp_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_cp.columns[rfecv.support_]))
    #%%
    #PASS TO RECURSIVE FEATURE EXTRACTION BANK PANEL
    #build a logistic regression and use recursive feature elimination to exclude trivial features
    #create the RFE model and select the eight most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_bp_train, y_bp_train)
    #selected attributes
    print('Selected features: %s' % list(X_bp_train.columns[rfe.support_]))
    print(rfe.ranking_)

    #Use the Cross-Validation function of the RFE module
    #accuracy is a classification metric!
    #r2 is a regression metric
    #accuracy describes the number of correct classifications
    rfecv = RFECV(estimator = log_reg, step = 1, cv = 8, scoring = 'accuracy')
    rfecv.fit(X_bp_train, y_bp_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_bp.columns[rfecv.support_]))

    #plot number of features VS. cross-validation scores
    #plt.figure(figsize = (10,6))
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #plt.show
'''
------------------------------------------------------------------------------
END Prediction for transaction_category_name and RFE for significant features
'''
#%%
'''
------------------------------------------------------------------------------
Prediction for student and RFE for significant features
INACTIVATED SINCE DATA IS UNLABELED YET
'''
'''
function to predict if the customer is a student or not
if the certainty is above 70% then more weight is given to these students
data is unlabeled and requires unsupervised learning algorithm
var set
x_cp
y_cp
x_bp
y_bp
split up values
'''
#use local outlier frequency
#append more weights in another column?
def predict_student():
    y_cp_student = df_card_rdy['student']
    X_cp_student = df_card_rdy[['amount', 'post_date_month', 'post_date_week',
        'post_date_weekday', 'file_created_date_month', 'transaction_category_name',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]

    try:
        y_cp_student.replace("nan", "unknown")
        y_cp_student.fillna(value = 'unknown')
    except:
        pass

    y_bp_student = df_bank_rdy['student']
    X_bp_student = df_bank_rdy[['amount', 'post_date_month', 'post_date_week',
        'transaction_category_name'
       'post_date_weekday', 'file_created_date_month',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]

    try:
        y_bp_student.replace("nan", "unknown")
        y_bp_student.fillna(value = 'unknown')
    except:
        pass

    X_cp_student_train, X_cp_student_test, y_cp_student_train, y_cp_student_test = train_test_split(X_cp_student, y_cp_student, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("CARD PANEL-STUDENT")
    print(f"Shape of the split training data set: X_cp_train:{X_cp_student_train.shape}")
    print(f"Shape of the split training data set: X_cp_test: {X_cp_student_test.shape}")
    print(f"Shape of the split training data set: y_cp_train: {y_cp_student_train.shape}")
    print(f"Shape of the split training data set: y_cp_test: {y_cp_student_test.shape}")
    #%%
    #TRAIN TEST SPLIT FOR BANK PANEL
    #from sklearn.model_selection import train_test_split
    X_bp_student_train, X_bp_student_test, y_bp_student_train, y_bp_student_test = train_test_split(X_bp_student, y_bp_student, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("BANK PANEL-STUDENT")
    print(f"Shape of the split training data set: X_bp_train:{X_bp_student_train.shape}")
    print(f"Shape of the split training data set: X_bp_test: {X_bp_student_test.shape}")
    print(f"Shape of the split training data set: y_bp_train: {y_bp_student_train.shape}")
    print(f"Shape of the split training data set: y_bp_test: {y_bp_student_test.shape}")

    log_reg = LogisticRegression(C = 0.01, class_weight = None, dual = False,
                               fit_intercept = True, intercept_scaling = 1,
                               l1_ratio = None, max_iter = 100,
                               multi_class = 'auto', n_jobs = None,
                               solver = 'lbfgs', tol = 0.0001, verbose = 0,
                               warm_start = False)
    #create the RFE model and select the eight most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_cp_student_train, y_cp_student_train)
    #selected attributes
    print('Selected features: %s' % list(X_cp_train.columns[rfe.support_]))
    print(rfe.ranking_)

    #Use the Cross-Validation function of the RFE module
    #accuracy describes the number of correct classifications
    rfecv = RFECV(estimator = log_reg, step = 1, cv = 8, scoring = 'accuracy')
    rfecv.fit(X_cp_student_train, y_cp_student_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_cp.columns[rfecv.support_]))
#%%
    log_reg = LogisticRegression(C = 0.01, class_weight = None, dual = False,
                               fit_intercept = True, intercept_scaling = 1,
                               l1_ratio = None, max_iter = 100,
                               multi_class = 'auto', n_jobs = None,
                               solver = 'lbfgs', tol = 0.0001, verbose = 0,
                               warm_start = False)
    #create the RFE model and select the eight most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_bp_student_train, y_bp_student_train)
    #selected attributes
    print('Selected features: %s' % list(X_bp_train.columns[rfe.support_]))
    print(rfe.ranking_)

    #Use the Cross-Validation function of the RFE module
    #accuracy describes the number of correct classifications
    rfecv = RFECV(estimator = log_reg, step = 1, cv = 8, scoring = 'accuracy')
    rfecv.fit(X_bp_student_train, y_bp_student_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_bp.columns[rfecv.support_]))

'''
------------------------------------------------------------------------------
END Prediction for transaction_category_name and RFE for significant features
'''
#%%
'''
------------------------------------------------------------------------------
Prediction for amount and RFE for significant features
predicting continuous values is impossible; so the amount will be encoded in brackets and
will form the label
'''
def predict_amount():
        le = LabelEncoder()
    try:
        if df_card_rdy['amount'].dtype == 'float64' or df_card_rdy['amount'].dtype == 'int64':
            df_card_rdy['amount_brackets'] = le.fit_transform(df_card_rdy['amount'])
        if df_bank_rdy['amount'].dtype == 'float64' or df_bank_rdy['amount'].dtype == 'int64':
                df_bank_rdy['amount_brackets'] = le.fit_transform(df_bank_rdy['amount'])
    except:
        raise Warning('column amount has not been converted to brackets!')

    y_cp_amount = df_card_rdy['amount_brackets']
    X_cp_amount = df_card_rdy[['post_date_month', 'post_date_week',
       'post_date_weekday',  'optimized_transaction_date_week',
       'file_created_date_month', 'transaction_category_name',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]
    try:
        y_cp_amount.replace("nan", "unknown")
        y_cp_amount.fillna(value = 'unknown')
    except:
        pass
    y_bp_amount = df_bank_rdy['amount_brackets']
    X_bp_amount = df_bank_rdy[['post_date_month', 'post_date_week',
       'post_date_weekday', 'file_created_date_month', 'transaction_category_name',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]
    try:
        y_bp_amount.replace("nan", "unknown")
        y_bp_amount.fillna(value = 'unknown')
    except:
        pass
    #APPLY THE SCALER FIRST AND THEN SPLIT INTO TEST AND TRAINING
    #PASS TO STANDARD SCALER TO PREPROCESS FOR PCA
    #ONLY APPLY SCALING TO X!!!
    scaler = StandardScaler()
    #fit_transform also separately callable; but this one is more time-efficient
    X_cp_amount_scl = scaler.fit_transform(X_cp_amount)
    X_bp_amount_scl = scaler.fit_transform(X_bp_amount)

    X_cp_amount_train, X_cp_amount_test, y_cp_amount_train, y_cp_amount_test = train_test_split(X_cp_amount_scl, y_cp_amount, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("CARD PANEL-AMOUNT")
    print(f"Shape of the split training data set: X_cp_train:{X_cp_amount_train.shape}")
    print(f"Shape of the split training data set: X_cp_test: {X_cp_amount_test.shape}")
    print(f"Shape of the split training data set: y_cp_train: {y_cp_amount_train.shape}")
    print(f"Shape of the split training data set: y_cp_test: {y_cp_amount_test.shape}")
    #%%
    #works with unscaled data already
    #TRAIN TEST SPLIT FOR BANK PANEL
    #from sklearn.model_selection import train_test_split
    X_bp_amount_train, X_bp_amount_test, y_bp_amount_train, y_bp_amount_test = train_test_split(X_bp_amount_scl, y_bp_amount, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("BANK PANEL-AMOUNT")
    print(f"Shape of the split training data set: X_bp_train:{X_bp_amount_train.shape}")
    print(f"Shape of the split training data set: X_bp_test: {X_bp_amount_test.shape}")
    print(f"Shape of the split training data set: y_bp_train: {y_bp_amount_train.shape}")
    print(f"Shape of the split training data set: y_bp_test: {y_bp_amount_test.shape}")

    log_reg = LogisticRegression(C = 0.01, class_weight = None, dual = False,
                               fit_intercept = True, intercept_scaling = 1,
                               l1_ratio = None, max_iter = 100,
                               multi_class = 'auto', n_jobs = None,
                               solver = 'lbfgs', tol = 0.0001, verbose = 0,
                               warm_start = False)
    #create the RFE model and select the eight most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_cp_amount_train, y_cp_amount_train)
    #selected attributes
    #print('Selected features: %s' % list(X_cp_amount_train.columns[rfe.support_]))
    print(rfe.ranking_)

    #Use the Cross-Validation function of the RFE module
    #accuracy describes the number of correct classifications
    rfecv = RFECV(estimator = log_reg, step = 1, cv = 8, scoring = 'accuracy')
    rfecv.fit(X_cp_amount_train, y_cp_amount_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_cp_amount.columns[rfecv.support_]))
#%%
    #PASS TO RECURSIVE FEATURE EXTRACTION BANK PANEL
    #build a logistic regression and use recursive feature elimination to exclude trivial features
    log_reg = LogisticRegression(C = 0.01, class_weight = None, dual = False,
                                   fit_intercept = True, intercept_scaling = 1,
                                   l1_ratio = None, max_iter = 100,
                                   multi_class = 'auto', n_jobs = None,
                                   penalty = 'l2', random_state = None,
                                   solver = 'lbfgs', tol = 0.0001, verbose = 0,
                                   warm_start = False)
    #create the RFE model and select the eight most striking attributes
    rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
    rfe = rfe.fit(X_bp_amount_train, y_bp_amount_train)
    #selected attributes
    print('Selected features: %s' % list(X_bp_amount_train.columns[rfe.support_]))
    print(rfe.ranking_)

    #Use the Cross-Validation function of the RFE module
    #accuracy describes the number of correct classifications
    rfecv = RFECV(estimator = log_reg, step = 1, cv = 8, scoring = 'accuracy')
    rfecv.fit(X_bp_amount_train, y_bp_amount_train)

    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % list(X_bp_amount.columns[rfecv.support_]))

    #plot number of features VS. cross-validation scores
    #plt.figure(figsize = (10,6))
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #plt.show
#%%
#WORKS!
def predict_city():
#3/24/2020: preprocessing module works and this is potentially obsolete
    #        le = LabelEncoder()
#try:
#if df_card_rdy['city'].dtype == 'object':
#    df_card_rdy['city'] = le.fit_transform(df_card_rdy['city'])
#if df_bank_rdy['city'].dtype == 'object':
#    df_bank_rdy['city'] = le.fit_transform(df_bank_rdy['city'])
#if df_card_rdy['city'].dtype == 'object':
#    df_card_rdy['city'] = le.fit_transform(df_card_rdy['city'])
#if df_bank_rdy['state'].dtype == 'object':
#    df_bank_rdy['state'] = le.fit_transform(df_bank_rdy['state'])
#except:
#    raise Warning('column city has not been converted to brackets! or is already converted')

    y_cp_city = df_card_rdy['city']
    X_cp_city = df_card_rdy[['post_date_month', 'post_date_week',
       'post_date_weekday',  'optimized_transaction_date_week',
       'file_created_date_month', 'transaction_category_name',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]
    try:
        y_cp_city.replace("nan", "unknown")
        y_cp_city.fillna(value = 'unknown')
    except:
        pass
    y_bp_city = df_bank_rdy['city']
    X_bp_city = df_bank_rdy[['post_date_month', 'post_date_week',
       'post_date_weekday', 'file_created_date_month', 'transaction_category_name',
       'file_created_date_week', 'file_created_date_weekday',
       'optimized_transaction_date_month', 'optimized_transaction_date_week',
       'optimized_transaction_date_weekday', 'swipe_date_month',
       'swipe_date_week', 'swipe_date_weekday',
       'panel_file_created_date_month', 'panel_file_created_date_week',
       'panel_file_created_date_weekday', 'amount_mean_lag3',
       'amount_mean_lag7', 'amount_mean_lag30', 'amount_std_lag3',
       'amount_std_lag7', 'amount_std_lag30']]
    try:
        y_bp_city.replace("nan", "unknown")
        y_bp_city.fillna(value = 'unknown')
    except:
        pass

    #TRAIN SPLIT FOR THE CARD PANEL
    X_cp_city_train, X_cp_city_test, y_cp_city_train, y_cp_city_test = train_test_split(X_cp_city, y_cp_city, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("CARD PANEL-city")
    print(f"Shape of the split training data set: X_cp_train:{X_cp_city_train.shape}")
    print(f"Shape of the split training data set: X_cp_test: {X_cp_city_test.shape}")
    print(f"Shape of the split training data set: y_cp_train: {y_cp_city_train.shape}")
    print(f"Shape of the split training data set: y_cp_test: {y_cp_city_test.shape}")
    #%%
    #TRAIN TEST SPLIT FOR BANK PANEL
    X_bp_city_train, X_bp_city_test, y_bp_city_train, y_bp_city_test = train_test_split(X_bp_city, y_bp_city, test_size = 0.3, random_state = 42)
    #shape of the splits:
    ##features: X:[n_samples, n_features]
    ##label: y: [n_samples]
    print("BANK PANEL-city")
    print(f"Shape of the split training data set: X_bp_train:{X_bp_city_train.shape}")
    print(f"Shape of the split training data set: X_bp_test: {X_bp_city_test.shape}")
    print(f"Shape of the split training data set: y_bp_train: {y_bp_city_train.shape}")
    print(f"Shape of the split training data set: y_bp_test: {y_bp_city_test.shape}")
#%%
'''
Predict the city with a Random Forest Regressor grid CARD PANEL
'''
RFR = RandomForestRegressor()
parameters = {'n_estimators': [5, 10, 100, 250, 300],
              'max_depth': [5, 10, 15, 25, 30],
              #'criterion': ['mse'],
              #'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 7,9 ,12 ]
             }
#Run the grid search
grid_obj = GridSearchCV(RFR, parameters,
#Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
                        cv = 5,
                        n_jobs = -1,
                        verbose = 1)
grid_obj = grid_obj.fit(X_cp_city_train, y_cp_city_train)
#Set the clf to the best combination of parameters
RFR = grid_obj.best_estimator_
#Fit the best algorithm to the regular data
RFR.fit(X_cp_city, y_cp_city)


predictions = RFR.predict(X_cp_city_test)
#if we want to re-scale, use these lines of code :
#predictions = predictions * (max_train - min_train) + min_train
#y_validation_RF = y_validation * (max_train - min_train) + min_train
#if not, keep this one:
y_validation_RF = y_cp_city_test
print('R2 score = ',r2_score(y_validation_RF, predictions), '/ 1.0')
print('MSE score = ',mean_squared_error(y_validation_RF, predictions), '/ 0.0')
#%%
#local outlier frequency
#Contamination to match outlier frequency in ground_truth
preds = lof(
  contamination=np.mean(ground_truth == -1.0)).fit_predict(X_cp_city)
#Print the confusion matrix
print(confusion_matrix(ground_truth, preds))

#anti-fraud system + labeling
#pick parameters to spot outliers in

#while loop to stop as soon s first income is hit and add upp income/expense

#pass this to flask and app to inject this initial balance

