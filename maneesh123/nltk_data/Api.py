import re
import os
import csv
from datetime import datetime, timedelta
import atexit
import base64
import pandas as pd
import nltk
import string
import time
import shutil
import logging
import configparser
import warnings
from string import digits
from logging.handlers import RotatingFileHandler
from datetime import datetime,timedelta
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

Date = '29-11-2019'
Ver  = 'ML_DataAnalyser1.0'
valid_characters = "*#@abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
LogCreationPath = ""
TrainingData_path = ""

def setup_locallogger():
    LOG_FILENAME = LogCreationPath
    formatter = logging.Formatter('[%(asctime)s|%(levelname)s:%(lineno)d]%(message)s')
    handler = logging.handlers.TimedRotatingFileHandler(LOG_FILENAME, when="h", interval=1, backupCount=144)
    handler.setFormatter(formatter)
    logger = logging.getLogger() # or pass string to give it a name
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    return logger

try:
    print("Reading Config File....")
    config = configparser.ConfigParser()
    config.readfp(open(r'System.ini'))
    print("__________________________________________________")
    print("ML_SpamModule Released on:",Date)
    print("Version:",Ver)
    print("Configuration file : System.ini")
    print("__________________________________________________")
    #load_Configuration
    Testingsourcepath = config.get('ML_DATA_ANALYSER', 'TESTING_DATA_PATH')
    resultpath =config.get('ML_DATA_ANALYSER', 'RESULT_PATH')
    processed_path = config.get('ML_DATA_ANALYSER', 'PROCESSED_DATA_PATH')
    TrainingData_path = config.get('ML_DATA_ANALYSER', 'TRAINING_DATA_PATH')
    Log_path = config.get('ML_DATA_ANALYSER', 'LOG_PATH')
    Log_tag = config.get('ML_DATA_ANALYSER', 'LOG_TAG')
    nltk_path = config.get('ML_DATA_ANALYSER', 'NLTK_PATH')
    Spam_File_enable = config.get('ML_DATA_ANALYSER', 'SPAM_FILE_ENABLE')
    LogCreationPath = Log_path+Log_tag
    print(Testingsourcepath)
    print(resultpath)
    print(processed_path)
    print(TrainingData_path)
    print(LogCreationPath)
    print(nltk_path)
    print(Spam_File_enable)
    setup_locallogger()
    nltk.data.path.append(nltk_path)
    print(nltk.data.path)
    print("In Training Model....")
    logging.critical(f'TrainingDataPath:{TrainingData_path}')
    logging.critical(f'TestingDataPath:{Testingsourcepath}')
    logging.critical(f'ResultFilePath:{resultpath}')
    logging.critical(f'ProcessedFilepath:{processed_path}')
    logging.critical(f'LogCreationPath:{LogCreationPath}')
    logging.critical(f'NltkPath:{nltk_path}{nltk.data.path}')
    logging.critical(f'Spam_File_enable:{Spam_File_enable}')
    training_df = pd.read_csv(TrainingData_path,encoding='iso-8859-1',names=['labels','training_messages'],engine='c')
    logging.critical('Loading Training Data successfull....')
    training_df['training_messages'] = training_df['training_messages'].apply(text_process)
    bow_transformer = CountVectorizer(analyzer=text_process).fit(training_df['training_messages'])
    messages_bow = bow_transformer.transform(training_df['training_messages'])
    tfidf_transformer=TfidfTransformer().fit(messages_bow)
    messages_tfidf=tfidf_transformer.transform(messages_bow)
    spam_detect_model = RandomForestClassifier(n_estimators=100).fit(messages_tfidf,training_df['labels'])
    print("ML Model is Trained successfully.....")
    logging.critical('ML Model is Trained successfully.....')
except Exception as exception:
    print("Exception in training model",exception)
    logging.error(f'Exception in training model:{exception}')


def text_process(mess):
    try:
        for char in str(mess):
            if char not in valid_characters:
                mess = str(mess).replace(char,' ')
        nopunc=''.join(mess)
        return [word.lower() for word in nopunc.split() if word not in stopwords.words('english') if len(word) != 1]
    except Exception as e:
        print("Exception in loading Text Process",e)
        logging.error(f'Exception in loading Text Process:{e}')

class Process(BaseHTTPRequestHandler):
    #global TrainingData_path
    global training_df
    global bow_transformer
    global tfidf_transformer
    global spam_detect_model
    global nltk_path
    def text_process(self,mess):
        try:
            for char in str(mess):
                if char not in valid_characters:
                    mess = str(mess).replace(char,' ')
            nopunc=''.join(mess)
            return [word.lower() for word in nopunc.split() if word not in stopwords.words('english') if len(word) != 1]
        except Exception as e:
            print("Exception in loading Text Process",e)
            logging.error(f'Exception in loading Text Process:{e}')
 
    def process_Predict(self,msg):
            #global bow_transformer
            global tfidf_transformer
            global spam_detect_model
            try:
                msg = self.text_process(msg)
                print("msg")
                df_test = pd.DataFrame({'test_data':[msg]})
                messages_bow_predict = bow_transformer.transform(df_test['test_data'])
                #TDIDF
                messages_tfidf_predict = tfidf_transformer.transform(messages_bow_predict)
                #MODEL PREDICTION
                logging.error(f' Started Processing >>>>>>>>>>>>>>>>>>')
                prediction = spam_detect_model.predict(messages_tfidf_predict)
                print("result is :",prediction)
                logging.error(f' Procesing for a record is completed')
                return prediction
            except Exception as exception:
                print("Exception in process_Predict",exception)    
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        """my_dict = {
           'foo': 42,
           'baz': "Hello",
           'poo': 124.2
        }"""
        my_dict = {
	"result": "ham",
	"detection": [{
		"spam": "40%"}, {
		"ham": "60%%"
	}]}
        #data= "sindu"
        my_json = json.dumps(my_dict, indent=2)
        self._set_response()
        self.wfile.write(bytes(my_json,'UTF-8'))
    
    def do_POST(self):
        #print(training_df.head())
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length).decode('utf-8')
        print("data before is:",post_data)
        message = json.loads(post_data) if post_data else None
        print("json is ",message)
        print("type is",message['msg'])
        msg = message['msg']
        val = self.process_Predict(msg) 
        print("print here", val)
        self.do_GET()

def run(server_class=HTTPServer, handler_class=Process, port=8080):
    #logging.basicConfig(level=logging.INFO)
    global LogCreationPath
    global TrainingData_path
    try:
        print("Reading Config File....")
        config = configparser.ConfigParser()
        config.readfp(open(r'System.ini'))
        print("__________________________________________________")
        print("ML_SpamModule Released on:",Date)
        print("Version:",Ver)
        print("Configuration file : System.ini")
        print("__________________________________________________")
        #load_Configuration
        Testingsourcepath = config.get('ML_DATA_ANALYSER', 'TESTING_DATA_PATH')
        resultpath =config.get('ML_DATA_ANALYSER', 'RESULT_PATH')
        processed_path = config.get('ML_DATA_ANALYSER', 'PROCESSED_DATA_PATH')
        TrainingData_path = config.get('ML_DATA_ANALYSER', 'TRAINING_DATA_PATH')
        Log_path = config.get('ML_DATA_ANALYSER', 'LOG_PATH')
        Log_tag = config.get('ML_DATA_ANALYSER', 'LOG_TAG')
        nltk_path = config.get('ML_DATA_ANALYSER', 'NLTK_PATH')
        Spam_File_enable = config.get('ML_DATA_ANALYSER', 'SPAM_FILE_ENABLE')
        LogCreationPath = Log_path+Log_tag
        print(Testingsourcepath)
        print(resultpath)
        print(processed_path)
        print(TrainingData_path)
        print(LogCreationPath)
        print(nltk_path)
        print(Spam_File_enable)
        setup_locallogger()
        nltk.data.path.append(nltk_path)
        print(nltk.data.path)
        server_address = ('', port)
        httpd = server_class(server_address, handler_class)
        logging.info('Starting httpd...\n')
        print("Starting httpd...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()
        logging.info('Stopping httpd...\n')      
        #httpd.serve_forever()
        
    except Exception as exception:
        print("Exception in training model",exception)
        logging.error(f'Exception in training model:{exception}')
    #httpd.server_close()
    #logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    #port = 9988
    run(port=int(9588))
    """from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()"""
