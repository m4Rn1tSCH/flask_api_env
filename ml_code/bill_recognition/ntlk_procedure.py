#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:49:03 2020

@author: bill
"""
import warnings
warnings.filterwarnings('ignore')
import os,sys,inspect
currentdir = os.path.abspath(inspect.getfile(inspect.currentframe()))
parentdir = os.path.abspath(os.path.dirname(currentdir))

project_dir = os.environ['PWD']
src_dir = project_dir + '/helpers'
sql_dir = project_dir + '/sql'

sys.path.insert(0,project_dir)


from helpers.postgres_helpers import Postgres
from helpers.text_cleanup import Cleanup


postg = Postgres()
initial_cleanup = Cleanup()

import nltk
nltk.download('wordnet')

stop_words = ['i', 'I', 'im', 'ive', 'me', 'my', 'myself', 'we', 'our', 'ours',
       'ourselves', 'you', 'youre', 'youve', 'youll', 'youd', 'your',
       'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
       'she', 'shes', 'her', 'hers', 'herself', 'it', 'its', 'its',
       'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
       'which', 'who', 'whom', 'this', 'that', 'thatll', 'these', 'those',
       'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
       'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
       'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
       'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
       'into', 'through', 'during', 'before', 'after', 'above', 'below',
       'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
       'under', 'again', 'further', 'then', 'once', 'here', 'there',
       'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
       'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
       'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
       'can', 'will', 'just', 'don', 'dont', 'should', 'shouldve', 'now',
       'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'arent',
       'couldn', 'doesn', 'doesnt', 'hadn', 'hadnt', 'hasn', 'hasnt',
       'haven', 'havent', 'isn', 'isnt', 'ma', 'mm', 'hmm', 'mightn',
       'mightnt', 'mustn', 'mustnt', 'needn', 'neednt', 'shan', 'shant',
       'shouldn', 'shouldnt', 'wasn', 'wasnt', 'weren', 'werent', 'won',
       'wont', 'wouldn', 'wouldnt', 'agent', 'mozilla', 'windows', 'nt',
       'gecko', 'oneagent', 'user', 'www', 'co', 'za', 'linux', 'android',
       'applewebkit', 'version', 'chrome', 'safari', 'a', 'b', 'c', 'd',
       'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
       'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'same', 'yours',
       'shes', 'any', 'be', 'youre', 'had', 'him', 'its', 'ma', 'which',
       't', 'out', 'o', 'if', 'his', 'both', 'theirs', 'nor', 'those',
       'arent', 'does', 'mustn', 'other', 'there', 'shant', 'i', 'mightn',
       'but', 'wasnt', 'me', 'neednt', 'up', 'should', 'hadn',
       'themselves', 'at', 'their', 'isn', 'below', 'your', 'werent',
       'we', 'youve', 'above', 'own', 'just', 'during', 'being',
       'have', 'our', 'into', 'all', 'some', 'for', 'shouldnt', 'couldn',
       'havent', 'wasn', 'wouldn', 'didnt', 'haven', 'weren', 'while',
       'youll', 'down', 'what', 'only', 'youd', 'most', 'now', 'between',
       'when', 'been', 'that', 'with', 'how', 'ourselves', 'than', 'ain',
       'don', 'on', 'not', 'no', 'll', 'wont', 're', 'who', 've', 'ours',
       'here', 'herself', 's', 'thatll', 'by', 'few', 'doesnt', 'they',
       'because', 'it', 'she', 'yourself', 'under', 'needn', 'you',
       'shan', 'whom', 'm', 'off', 'after', 'her', 'or', 'shouldve',
       'was', 'these', 'of', 'such', 'isnt', 'is', 'through', 'then',
       'itself', 'himself', 'mustnt', 'until', 'doesn', 'hadnt', 'the',
       'against', 'won', 'has', 'once', 'were', 'a', 'hasn', 'hasnt',
       'myself', 'having', 'each', 'more', 'do', 'did', 'this', 'my',
       'doing', 'about', 'why', 'its', 'shouldn', 'couldnt', 'are',
       'before', 'as', 'in', 'wouldnt', 'yourselves', 'over', 'didn',
       'aren', 'and', 'mightnt', 'he', 'dont', 'd', 'again', 'to',
       'further', 'hers', 'where', 'very', 'too', 'them', 'am', 'so',
       'an', 'from', 'will', 'can']

def strip(text):
    text = text.strip()
    return text

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def remove_stop_words_and_stem(sentence):
    tokens = sentence.split()
    lemmatizer = WordNetLemmatizer()
    sno = nltk.stem.SnowballStemmer('english')
    
    filtered_sentence = [token for token in tokens if not token in stop_words]
    pre_final = [sno.stem(token) for token in filtered_sentence]
    pre_final = [lemmatizer.lemmatize(token, pos="v") for token in pre_final]
    final = [token for token in pre_final if token.isalpha()]
   
    return ' '.join(final)

dataframe = postg.query(file_name=sql_dir+'/text_cleanup/bank_transact_data.sql')
cleaned_data = initial_cleanup.Clean_Text(Vendor_list=dataframe)
cleaned_data_1 = cleaned_data[['description','Merchant_noState']].copy()
cleaned_data_1['ic_description_stripped'] = cleaned_data['Merchant_noState'].apply(lambda x:
                                                                             strip(x)
                                                                         )


del cleaned_data_1['Merchant_noState']


cleaned_data_1['ic_description_nltk'] = cleaned_data_1['ic_description_stripped'].apply(lambda x:
                                                                             remove_stop_words_and_stem(str(x))
                                                                         )
    

del cleaned_data_1['ic_description_stripped']

