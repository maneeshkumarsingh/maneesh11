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

try:

    with open('A2P_P2P_Config.json') as config_file:
        config = json.load(config_file)

    #logging_streamlit = config['STREAMLIT_LOG_PATH']

    vectorizer_save = config['VECTORIZER']
    model_save = config['MODEL']
    logpath = config['LOG_PATH']
    #P2P_File_enable = config['PROCESS_FILE_EXTENSION']

    vectorizer = pickle.load(open(vectorizer_save, 'rb'))
    A2P_P2P_detect_model = pickle.load(open(model_save, 'rb'))
    logging.basicConfig(filename=logpath, level=logging.INFO, format='%(asctime)s|%(levelname)s:%(lineno)d]%(message)s')
    # Streamlit Web App
    # title of page
    st.title('A2P_P2P Prediction')
    print('pickle loaded')



    text = st.text_input('Please Enter Text')
    pressed = st.button('Submit')

    # url based checking
    if pressed:
        pred = A2P_P2P_detect_model.predict(vectorizer.transform([text]))[0]
        if pred == 'A2P':
            # st.write('Phished Url')
            st.markdown("<h1 style='text-align: left; color: black;'> A2P</h1>" , unsafe_allow_html=True)
            # st.markdown('**Phished Url**')
            # st.image('phis.png', width=225)
        else:
            st.markdown("<h1 style='text-align: left; color: black;'> P2P</h1>" , unsafe_allow_html=True)
            # textColor = '#f90000'
            # st.markdown('**Safe Url**')
            # st.write('Safe Url')


except:
    logging.info('Error in Streamlit file')
