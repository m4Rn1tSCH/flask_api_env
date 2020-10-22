#############ML logic for internal beta period#########
#INCOMING TRANSACTION
'''
Incoming approved transaction transmitted by the Q2 API in the Q2-format
'''

import pandas as pd
import os
import re
#%%
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
######LOADING THE TRANSACTION FILES#####
transaction_file = r"C:\Users\bill-\OneDrive - Education First\Documents\Docs Bill\FILES_ENVEL\Q2_test_data.csv"
path_1 = transaction_file.replace(os.sep,'/')
transactions = ''.join(('', path_1, ''))
'''
SCRIPT WILL GET ALL CSV FILES AT THIS STAGE!
'''
#relative_t_path = './*.csv'
df = pd.read_csv(transactions, index_col = [0])
#%%
#vendor_file = r"C:\Users\bill-\Dropbox\Nan\Archived\BillVendors_Only.xlsx"
#path_11 = vendor_file.replace(os.sep,'/')
#vendors = ''.join(('', path_11, ''))
#relative_v_path = './BillVendors_Only.xlsx'
#%%
###connection to flask with relative link#####
###FLASK FUNCTION###
#from flask import Flask
#############SETTING UP THE APP##########
#app = Flask(__name__)
####TRIGGER URL#####
#@app.route('/')

#RUN THE APPLICATION
#flask command or -m swith in Python

########SETTING THE ENVIRONMENT VARIABLE#######
#$ export FLASK_APP=CONNECTION.py
#$ flask run
#* Running on http://127.0.0.1:5000/

####COMMAND PROMPT#####
#C:\path\to\app>set FLASK_APP=CONNECTION.py

####for production use##
#$ flask run --host=0.0.0.0
#%%
#CHECK FOR TRANSACTION HISTORY
'''
If previous transactions are in the list/database/history, it is passed to check for missing values
If there are no precedent transactions, the object will be passed to the income splitter
'''
#%%
#PASS TO NEUTRAL SPLITTING ALGORITHM
import One_function_trial_pure.py as split_eng
#######
df = df
income = 1000
newincome = 1300
bills = 250
bill_change_value = 30
#######
df_split = split_eng.one_trial(df, income, newincome, bills, bill_change_value)
'''
Paavna's splitting algorithm will apply an initial split without the involvement of AI or a dynamic element
'''
#%%
#CHECK FOR MISSING VALUES
'''
find missing values and mark the corresponding column as target that is to be predicted
'''
#iterate over all columns and search for missing values
#find such missing values and declare it the target value
#df in use is pandas datafame, use .iloc[]
#df is a dictionary, .get()
#iterate over columns first to find missing targets
#iterate over rows of the specific column that has missing values
#fill the missing values with a value
target_list = []
for col in df.columns:
    if df[col].isnull().any() == True:
        print(f"{col} is target variable and will be used for prediction")
        target_list.append(col)
        for index, row in df.iterrows():
            if row.isnull() == True:
                print(f"Value missing in row {index}")
                df[row].fillna(method = bfill)
            else:
                print("Data contains no missing values; specify the label manually")
                pass
            #df.iloc[lambda df: [df[col.isnull() == True]], column]
            #df.iterrows
#%%
##V2
#first .all() for the entire column; second .all() for the entire df as a single value
#assert df.iloc[:, 1:].apply(df.notnull(), axis=1).all().all()
#%%
#######FIX IMPUTER SOLUTION
#V3
#not running with anaconda python 3.7.3
#use imputer from sklearn
#axis = 0 col; axis = 1 row
#from sklearn. import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

#%%
#LABEL ENCODER
'''
encode all non-numerical values to ready up the data set for classification and regression purposes
'''
#applying fit_transform yields: encoding of 22 columns but most of them remain int32 or int64
#applying first fit to train the data and then apply transform will encode only 11 columns and leaves the others unchanged
#if 2 or fewer unique categories data type changes to "object"
#iterate through columns and change the object (not int64)

###ATTENTION; WHEN THE DATA FRAME IS OPEN AND THE SCRIPT IS RUN
###THE DATA TYPES CHANGE TO FLOAT64 AS THE NUMBERS ARE BEING DISPLAYED
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0
#           V1
#Iterate through the columns
#Train on the training data
#Transform both training and testing
#Keep track of how many columns were converted
for col in df:
    if df[col].dtype == 'object':
        le.fit(df[col])
        df[col] = le.transform(df[col])
        le_count += 1

print('%d columns were converted.' % le_count)

#for comparison of the old data frame and the new one
print("PROCESSED DATA FRAME:")
print(df.head(3))
print("new data frame ready for use")
#%%
#SET UP LABELS WHICH ARE BEING PREDICTED BY FEATURES
#set the features
a = input(f"choose from following columns of given dataframe: {df.columns}; type in the desired label to be predicted...")
X = df.pop(index = picked_feat.index(a))
#set the label
y = picked_feat
#%%
#PASS TO STANDARD SCALER TO PREPROCESS FOR PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# calling fit and transform in sequence (using method chaining)
#X_scaled = scaler.fit(X).transform(X)
# same result, but more efficient computation
X_scaled_d = scaler.fit_transform(X)
#%%
#PASS TO PRINCIPAL COMPONENT ANALYSIS
from sklearn.decomposition import PCA
#keep the most important features of the data
#in this case; keep all columns that have not been picked by the algorithm
pca = PCA(n_components = int(len(df.columns) - 1))
#fit PCA model to breast cancer data
pca.fit(X_scaled)
#transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))
#%%
#TRAIN TEST SPLIT INTO TWO DIFFERENT DATASETS - SCALED
#Train Size: 50% of the data set
#Test Size: remaining 50%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.3, random_state = 42)
#shape of the splits:
##features: X:[n_samples, n_features]
##label: y: [n_samples]
print(f"Shape of the split training data set: X_train:{X_train.shape}")
print(f"Shape of the split training data set: X_test: {X_test.shape}")
print(f"Shape of the split training data set: y_train: {y_train.shape}")
print(f"Shape of the split training data set: y_test: {y_test.shape}")
#%%
#TRAIN TEST SPLIT INTO TWO DIFFERENT DATASETS - UNSCALED
#Train Size: 50% of the data set
#Test Size: remaining 50%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#shape of the splits:
##features: X:[n_samples, n_features]
##label: y: [n_samples]
print(f"Shape of the split training data set: X_train:{X_train.shape}")
print(f"Shape of the split training data set: X_test: {X_test.shape}")
print(f"Shape of the split training data set: y_train: {y_train.shape}")
print(f"Shape of the split training data set: y_test: {y_test.shape}")
#%%
#PASS TO RECURSIVE FEATURE EXTRACTION
'''
all other columns are features and need to be checked for significance to be added to the feature list
'''
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

#Creating training and testing data
train = df.sample(frac = 0.5, random_state = 200)
test = df.drop(train.index)

#pick feature columns to predict the label
#y_train/test is the target label that is to be predicted
#PICKED LABEL = FICO numeric
cols = ["type", "amount", "isCredit", "returnCode", "feeCode", "subTypeCode", "subType", "check", "Student", "account_balance", "Age", "CS_FICO_str", "CS_internal"]
X_train = train[cols]
y_train = train['CS_FICO_num']
X_test = test[cols]
y_test = test['CS_FICO_num']
#build a logistic regression and use recursive feature elimination to exclude trivial features
log_reg = LogisticRegression()
#create the RFE model and select the eight most striking attributes
rfe = RFE(estimator = log_reg, n_features_to_select = 8, step = 1)
rfe = rfe.fit(X_train, y_train)
#selected attributes
print('Selected features: %s' % list(X_train.columns[rfe.support_]))
print(rfe.ranking_)

#Use the Cross-Validation function of the RFE module
#accuracy describes the number of correct classifications
rfecv = RFECV(estimator = LogisticRegression(), step = 1, cv = 8, scoring = 'accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

#plot number of features VS. cross-validation scores
#plt.figure(figsize = (10,6))
#plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (nb of correct classifications)")
#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()
#%%
#PASS TO ML ALGO
#RANDOM FOREST MODULE
'''
use the random forest regressor algorithm to predict labels
'''
#Packages
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(n_estimators = 150, max_depth = len(df.columns), min_samples_split = 5)
RFR.fit(X_train, y_train)
y_pred = RFR.predict(X_test)
f"Training set accuracy: {RFR.score(X_train, y_train)}; Test set accuracy: {RFR.score(X_test, y_test)}; Test set validation: {RFR.score(X_test, y_pred)}"
#%%
#SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
#function of Select K Best is here f_classifier; determines features based on the f-values between features & labels
#other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_classif
#RandomForestClassifier is insufficient and does not provide enough splits
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#Create pipeline with feature selector and classifier
#replace with gradient boosted at this point or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERS NEED TO HAVE THE SAME LENGTH
params = {
   'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
   'clf__n_estimators':[10, 25, 45, 100, 150]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
#%%
#GRADIENT BOOSTING DECISION TREE
'''
use the gradient boosting decision to predict labels
'''
#SelectKBest picks features based on their f-value to find the features that can optimally predict the labels
#funtion of Selecr K Best is here f_classifier; determines features based on the f-values between features & labels
#other functions: mutual_info_classif; chi2, f_regression; mutual_info_regression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#Create pipeline with feature selector and classifier
#replace with gradient boosted at this point or regressor
pipe = Pipeline([
    ('feature_selection', SelectKBest(score_func = f_classif)),
    ('clf', GradientBoostingClassifier(random_state = 42))])

#Create a parameter grid
#parameter grids provide the values for the models to try
#PARAMETERs NEED TO HAVE THE SAME LENGTH
params = {
   'feature_selection__k':[1, 2, 3, 4, 5, 6, 7],
   'clf__n_estimators':[15, 25, 50, 75, 120, 200, 350]}

#Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid = params)

#Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)
##RESULT
#the labels only provide one member per class, that makes the current data set
#unsuitable for a pickle file
#%%
#KNN
'''
use the K-nearest neighbor to predict labels
'''
#K NEAREST NEIGHBOR CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
metric_params = None, n_jobs = 1, n_neighbors = 1, p = 2, weights = 'uniform')
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
f"Training set accuracy: {KNN.score(X_train, y_train)}; Test set accuracy: {KNN.score(X_test, y_test)}; Test set validation: {KNN.score(X_test, y_pred)}"
#%%
#MLP NEURAL NETWORK
'''
use a multi-layered perceptron to predict labels
'''
#NO GPU SUPPORT FOR SKLEARN
from sklearn.neural_network import MLPClassifier

#adam: all-round solver for data
#hidden_layer_sizes: no. of nodes/no. of hidden weights used to obtain final weights;
#match with input features
#alpha: regularization parameter that shrinks weights toward 0 (the greater the stricter)
MLP = MLPClassifier(hidden_layer_sizes = 250, solver = 'adam', alpha = 0.01 )
MLP.fit(X_train, y_train)
y_pred = MLP.predict(X_test)
f"Training set accuracy: {MLP.score(X_train, y_train)}; Test set accuracy: {MLP.score(X_test, y_test)}; Test set validation: {MLP.score(X_test, y_pred)}"
#%%
#MEASURE ACCURACY OF PREDICTION
#CREATE CONTAINER FOR ACCURACY METRICS
'''
pick accuracy measures
store the value in a data frame
'''
from sklearn.metrics import accuracy_score
accuracy_table = pd.DataFrame(data = {'Y_TEST_VALUE_SCORE':[KNN.score(X_test, y_test), RFC.score(X_test, y_test), RFR.score(X_test, y_test_rfr), Logreg.score(X_test, y_test_lgreg), MLP.score(X_test, y_test_mlp)],
                                      'Y_TEST_PREDICTION_SCORE':[KNN.score(X_test, y_pred_knn), RFC.score(X_test, y_pred_rfc), RFR.score(X_test, y_pred_rfr), Logreg.score(X_test, y_pred_lgreg), MLP.score(X_test, y_pred_mlp)],
                                      'AUC_VALUE':[auc_score(y_test, y_pred_knn), auc_score(y_test, y_pred_rfc), auc_score(y_test, y_pred_rfr), auc_score(y_test, y_pred_lgreg), auc_score(y_test, y_pred_mlp)],
                                      'CONFUSION_MATRIX':[confusion_matrix(y_test, y_pred_knn), confusion_matrix(y_test, y_pred_rfc), confusion_matrix(y_test, y_pred_rfr), confusion_matrix(y_test, y_pred_lgreg), confusion_matrix(y_test, y_pred_mlp)]})
#%%
#PASS PREDICTED VALUES TO APP
'''
let the app display the suggested value in a separate window and ask for confirmation
"correct" OR "wrong prediction"
'''
#%%
#AWAIT RESPONSE FROM APP
'''
await user input
'''
#%%
#ADJUST WEIGHTS AND APPEND TO CSV FILE FOR FUTURE TRAINING
'''
allocate more weight to correct predictions for later training
create a new column for the weights as numerical values or with a marker
these weights will give constellations of students conducting transactions more weights
'''
#dict or dataframe?
enhancement_weights = []
#%%
#WRITE THIS TRANSACTION IN ITS ENTIRETY TO A CSV ROW THAT IS THE TRAINING DATA
'''
write it on a per-line basis to the csv that will sit in the flask folder and will later be available for training
'''
with open('Proportions-Auto recovered.csv','a') as newFile:
    newFileWriter=csv.writer(newFile)
    newFileWriter.writerow([X1, X2, X3, X4, X5])
    return(X2, X3, X4, X5)
    print("row has been written to training data...")
    break

#ALTERNATIVE VERSION
#import pandas as pd
##WILL BE SAVED IN THE SAME FOLDER AS THIS SCRIPT
#Create a Pandas Excel writer using XlsxWriter as the engine.
#def export_df():
#    sheet_name = 'Sheet_1'
#    writer     = pd.ExcelWriter('filename.xlsx', engine='xlsxwriter')
#    df.to_excel(writer, sheet_name=sheet_name)

#Access the XlsxWriter workbook and worksheet objects from the dataframe.
#    workbook  = writer.book
#    worksheet = writer.sheets[sheet_name]

#Adjust the width of the first column to make the date values clearer.
#    worksheet.set_column('A:A', 20)

#Close the Pandas Excel writer and output the Excel file.
#    writer.save()