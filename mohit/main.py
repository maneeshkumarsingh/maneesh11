import streamlit as st
import pickle
import logging
import os
import time
import json
import pickle
import os
import csv
import pandas as pd
#from clean import clean_texts
#from utils import getAvgFeatureVecs,num_features
#from gensim.models import word2vec
#from predict import predict
pd.options.mode.chained_assignment = None
with open('Spam_Ham_Config.json') as config_file:
    config = json.load(config_file)

# logging_streamlit = config['STREAMLIT_LOG_PATH']

vectorizer_save = config['VECTORIZER']
model_save = config['MODEL']
logpath = config['LOG_PATH']
# P2P_File_enable = config['PROCESS_FILE_EXTENSION']

vectorizer = pickle.load(open(vectorizer_save, 'rb'))
spam_ham_detect_model = pickle.load(open(model_save, 'rb'))
logging.basicConfig(filename=logpath, level=logging.INFO, format='%(asctime)s|%(levelname)s:%(lineno)d]%(message)s')
# Streamlit Web App
def process_Predict(chunk):
    global vectorizer
    global spam_ham_detect_model
    # global count
    try:
        messages_tfidf_predict = vectorizer.transform(chunk['text'])#.values.astype('U'))
        prediction = spam_ham_detect_model.predict(messages_tfidf_predict)
        ham_prob = spam_ham_detect_model.predict_proba(messages_tfidf_predict)[:, 0]
        spam_prob = spam_ham_detect_model.predict_proba(messages_tfidf_predict)[:, 1]
        chunk['Predicted'] = prediction
        return chunk

    except Exception as exception:
        print("Exception in process_Predict:", exception)
        #logging.error(f'Exception in process_Predict:{exception}')

def convert_df(df):
   return df.to_csv()#.encode('utf-8')

def app():

    st.title('Spam Ham Prediction')
    st.subheader("would you like to test with")
    single = st.checkbox('Single test')
    bulck = st.checkbox('Bulk data set')


    # url based checking
    if single:
        text = st.text_input('Please Enter Text')
        pressed = st.checkbox('Submit 👈')
        if pressed:
            #vectorizer = pickle.load(open(vectorizer_save, 'rb'))
            #spam_ham_detect_model = pickle.load(open(model_save, 'rb'))
            pred = spam_ham_detect_model.predict(vectorizer.transform([text]))[0]
            if pred == 'SPAM':
                st.markdown("<h1 style='text-align: left; color: black;'> SPAM</h1>", unsafe_allow_html=True)
            # st.markdown('**Phished Url**')
            # st.image('phis.png', width=225)
            else:
                st.markdown("<h1 style='text-align: left; color: black;'> HAM</h1>", unsafe_allow_html=True)
            # textColor = '#f90000'
            # st.markdown('**Safe Url**')
            # st.write('Safe Url')

    if bulck:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file, encoding='iso-8859-1', names=[ 'text'], engine='c')  # , encoding='unicode_escape')

            chunk_df = (process_Predict(new_df))
            csv = convert_df(chunk_df)

            st.download_button(
                "Press to Download",
                csv,
                "Predicted_file.csv",
                "text/csv",
                key='download-csv'
            )

            #new_df = pd.read_csv(uploaded_file, encoding='iso-8859-1', names=['labels', 'training_messages'], engine='c')  # , encoding='unicode_escape')


