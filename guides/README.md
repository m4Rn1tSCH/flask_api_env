# Envel Machine Learning

# Coding environment: spec_file.txt

# Take a look at the FLOWCHART to see the workflow
# WORKFLOW
1) Pull data from PostgreSQL Database with Transactions purchased from Yodlee; This database has the transaction data of 100,000 customer from all across the US on it. ("Yodlee")<br/>
1a) Yodlee Structure:<br/>
    Table with all demographic columns<br/>
    Table with Bank Transactions<br/>
    Table with Card Transactions<br/>
    ### Files:

2) Preprocess the data and make sure all columns have only numerical values
   Drop the currency column in case there is only one currency
   'optimal_transaction_date' is index column
   Fix the other date columns or drop them
   ### Files:
   
3) Split the data into test and training set<br/>
3a) Optional test/training parts available:<br/>
    Standard-scaled<br/>
    MinMax-scaled<br/>
    Processed with Principal Component Reduction<br/>
    Clustering with K-Means<br/>
3b) Targets<br/>
Target: City, in which the transaction took place; Features: All other columns<br/>
Target: State, in which the transaction took place; Features: All other columns<br/>
Target: amount_mean_lag7, in which the transaction took place; Features: All other columns<br/>
   ### Files: 

4) Feed to model and fit("train") the model<br/>
   ### Files:
4a) Pipelines with most Sklearn regressors and classifiers<br/>
   ### Files:
4b) Tensorflow 2<br/>
   ### Files:
5) Evaluation of accuracy of model<br/>
   ### Files:
6) Save the trained model as pickle file in the working directory<br/>
   ### Files:

# Naming conventions
Python_NN_[] - sklearn Neural Network<br/>
Python_NN_TF2_[] - Tensorflow 2 Neural Network<br/>
Python_AI_[] sklearn library<br/>

# FILES and their PURPOSE
CORE FILES - Files are finished and can be integrated or are integrated (flask function/executable)<br/>
 'PostgreSQL_credentials.py',<br/>
 'Python_AI_Pipeline_Classification.py',<br/>
 'Python_AI_Pipeline_NN.py',<br/>
 'Python_AI_Pipeline_NN_unified.py',<br/>
 'Python_AI_Pipeline_Regression.py',<br/>
 'Python_Basic_RNN_GRU_cell_regression.py',<br/>
 'Python_Basic_RNN_LSTM_cell_regression.py',<br/>
 'Python_inc_exp_bal_database.py',<br/>
 'Python_inc_exp_bal_database_dynamic.py',<br/>
 'Python_inc_exp_bal_database_static.py',<br/>
 'Python_injection_dbms_csv_exp_function_dynamic.py',<br/>
 'Python_injection_dbms_csv_exp_function_static.py',<br/>
 'Python_eda_ai.py',<br/>
 'Python_RNN_Regression.py',<br/>
 'Python_spending_report_csv_function.py',<br/>
 'Python_feat_engineering_script.py',<br/>
 'Python_SQL_connection.py',<br/>
 'Python_TF2_NN_regression.py',<br/>
 'Python_TF_LSTM_multivariate_regression.py',<br/>

# RAW SNIPPETS
 'Python_CSV_export_function.py',<br/>
 'Python_bill_list_recognition.py',<br/>
 'Python_bill_receipt_OCR_classifier_module.py',<br/>
 'Python_bill_recognition_flask_version_abs_path.py',<br/>
 'Python_bill_recognition_flask_version_rel_path.py',<br/>
 'Python_Classification_pipe_unified.py',<br/>
 'Python_Regression_pipe_unified.py',<br/>

# UNCLASSIFIED
 'Python_data_load_random_sample.py',<br/>
 'Python_df_label_encoding.py',<br/>
 'Python_feat_engineering_script_yodlee.py',<br/>
 'Python_file_reader_converter.py',<br/>
 'Python_NN_application.py',<br/>
 'Python_OCR_bill_recognition_snippet.py',<br/>
 'Python_prediction_merged_function_database.py',<br/>
 'Python_preprocessing_merged_function_database.py',<br/>
 'Python_random_sample.py',<br/>
 'Python_random_str_num_gen.py',<br/>
 'Python_rand_int_row_pick.py',<br/>
 'Python_Time_Series_Prediction.py'<br/>


