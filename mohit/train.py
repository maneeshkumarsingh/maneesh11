import pandas as pd
import re
import nltk
import string
import time
import shutil
import logging
import configparser
import warnings
from string import digits
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from nltk.corpus import stopwords
#from config import TrainingData, Add_TrainingData, model_save, vectorizer_save, Time_interval
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import json
with open('Spam_Ham_Config.json') as config_file:
    config = json.load(config_file)

trainingData = config['TRAINING_DATA_PATH']
#add_trainingData = config['Add_TrainingData']
vectorizer_save = config['VECTORIZER']
model_save = config['MODEL']

#Log_path = config['Model_configration']['LOG_PATH']

def text_process(mess):
    try:
        assert(type(mess) == str)
        cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', mess)
        cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', cleaned)
        cleaned = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr', cleaned)
        cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
        nopunc=''.join(cleaned)
        return ' '.join(word.lower() for word in nopunc.split() if word not in stopwords.words('english') if len(word) != 1)
    except Exception as e:
        print("Exception in loading Text Process", e)
        logging.error(f'Exception in loading Text Process:{e}')

def train():
    #global training_df
    try:
        training_df = pd.read_csv(trainingData, encoding='iso-8859-1', names=['labels', 'training_messages'], engine='c')
        #AddingData_df = pd.read_csv(add_trainingData, names=['labels', 'training_messages'])
        #training_df = trainingdf.append(AddingData_df, ignore_index=True)

        logging.info('Loading Training Data successfull....')
        print(('Loading Training Data successfull....'))

        training_df['labels'] = training_df['labels'].str.upper()
        training_df['training_messages'] = training_df['training_messages'].apply(text_process)
        print('Text_cleand')
        training_df["messages_Length"] = training_df["training_messages"].str.len()

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X_ngrams = vectorizer.fit_transform(training_df['training_messages'])

        A2P_P2P_detect_model = RandomForestClassifier(n_estimators=100).fit(X_ngrams, training_df['labels'])

        print("ML Model is Trained successfully.....")
        #logging.info('DB Connection successful')
        logging.info('ML Model is Trained successfully.....')


        # Save the vectorizer
        pickle.dump(vectorizer, open(vectorizer_save, 'wb'))
        # Saving Model
        pickle.dump(A2P_P2P_detect_model, open(model_save, 'wb'))

    except Exception as exception:
        print("Exception in training model", exception)
        logging.info(f'Exception in training model:{exception}')


if __name__ == '__main__':
    train()