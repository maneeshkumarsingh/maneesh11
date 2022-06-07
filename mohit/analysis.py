import streamlit as st
import seaborn as sns
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
import numpy as np
import json
import logging
pd.options.mode.chained_assignment = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report,confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

with open('Spam_Ham_Config.json') as config_file:
  config = json.load(config_file)

TrainingData_path = config["TRAINING_DATA_PATH"]

Log_path = config['LOG_PATH']

vectorizer_filename = config['TEST_VECTORIZER']
model_filename = config['TEST_MODEL']

nltk_path = config['NLTK_PATH']
logging.basicConfig(filename=Log_path, level=logging.INFO, format='%(asctime)s|%(levelname)s:%(lineno)d]%(message)s')

#vectorizer_filename = "vectorizer.pk"
#model_filename = "spam_ham_model.pk"

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


# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    A = (((C.T) / (C.sum(axis=1))).T)
    B = (C / C.sum(axis=0))
    plt.figure(figsize=(20, 4))
    labels = [1, 2]
    # representing A in heatmap format
    cmap = sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")

    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")

    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")

    plt.show()

def app():
    st.title('Check my learning IQ 🐼')
    st.subheader('My Existing Learning set 🐼')
    training_df1 = pd.read_csv(TrainingData_path, encoding='iso-8859-1', names=['labels', 'training_messages'], engine='c')
    st.write(training_df1)
    st.write('training data shape = ', training_df1.shape)
    st.write('Spam  and Ham Count in Training Data Set')
    training_df1['labels'] = training_df1['labels'].str.upper()
    st.write( training_df1.labels.value_counts())
    #st.write('HAM = ', training_df1.HAM.value_counts())
    st.subheader("Add more ideas to enhance my learning 🐼")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file, encoding='iso-8859-1', names=['labels', 'training_messages'], engine='c')  # , encoding='unicode_escape')

        st.write('Uploded data shape = ', new_df.shape)
        new_df['labels'] = new_df['labels'].str.upper()
        st.write(new_df)
        st.write('Spam and Ham Count in  Uploded Training Data Set :smiley:')
        st.write(new_df.labels.value_counts())
        training_df = pd.concat([training_df1, new_df], axis=0)
        training_df.reset_index(inplace=True)
        training_df = training_df[['labels', 'training_messages']]
        st.write('mearge data shape = ', training_df.shape)
        st.write(training_df)
        training_df['labels'] = training_df['labels'].str.upper()
        st.write('Spam and Ham Count in  Meagre Training Data Set :smiley:')
        st.write(training_df.labels.value_counts())
        # st.write('HAM = ', training_df.HAM.value_counts())
        st.write('Successfull mearge both Training Data :smiley: ')
        st.write('Start Text Cleaning and Vectorization :smiley:')

        training_df['training_messages'] = training_df['training_messages'].apply(text_process)

        vectorizer = TfidfVectorizer()  # max_features=2500, min_df=7, max_df=0.8)
        X_ngrams = vectorizer.fit_transform(training_df['training_messages'])
    #################################################
        x_train, x_test, y_train, y_test = train_test_split(X_ngrams, training_df['labels'], shuffle=True, test_size=0.2,
                                                            random_state=42)#, stratify=training_df['labels'])
    # spam_detect_model = RandomForestClassifier(n_estimators=100).fit(X_ngrams, training_df['labels'])
        st.write('Modal Training Start')
        model = RandomForestClassifier(n_estimators=250, random_state=0)
        model.fit(x_train, y_train)
        st.write("ML Model is Trained successfully :smiley:")
        # Save the vectorizer
        #pickle.dump(vectorizer, open(vectorizer_save, 'wb'))
        pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
        # Saving Model
        #pickle.dump(A2P_P2P_detect_model, open(model_save, 'wb'))
        pickle.dump(model, open(model_filename,'wb'))

        st.write("ML Model successfully save in pickle :smiley:")

        preds = model.predict(x_test)
        accuracy = accuracy_score(preds, y_test)
        st.write('accuracy = ', accuracy)
        st.write(plot_confusion_matrix(y_test, preds))
        st.write(classification_report(y_test, preds))
    #plot_confusion_matrix(val_x, preds)
        logging.critical('ML Model is Trained successfully :smiley:')


        vectorizer_model = pickle.load(open(vectorizer_filename, 'rb'))
        spam_ham_model = pickle.load(open(model_filename, 'rb'))

        # [text_proces(message)])):
        def spam_filter(message):
            if spam_ham_model.predict(vectorizer_model.transform(message)):
                return 'spam'
            else:
                return 'ham'

        st.subheader('Spam Ham detection')



        text = st.text_input('Please Enter Text')
        pressed = st.button('Submit')

    # url based checking
        if pressed:
            #pred = model.predict(vectorizer.transform([text]))[0]
            pred = model.predict(vectorizer.transform([text]))[0]
            if pred == 'SPAM':
                # st.write('Phished Url')
                st.markdown("<h1 style='text-align: left; color: black;'> SPAM</h1>", unsafe_allow_html=True)
                # st.markdown('**Phished Url**')
                # st.image('phis.png', width=225)
            else:
                st.markdown("<h1 style='text-align: left; color: black;'> HAM</h1>", unsafe_allow_html=True)
                # textColor = '#f90000'
                # st.markdown('**Safe Url**')
                # st.write('Safe Url')

app()



