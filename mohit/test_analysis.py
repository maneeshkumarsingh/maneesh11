import streamlit as stimport seaborn as snsimport pandas as pdimport reimport picklefrom nltk.corpus import stopwordsimport numpy as npimport jsonimport loggingpd.options.mode.chained_assignment = Nonefrom sklearn.ensemble import RandomForestClassifierfrom sklearn.metrics import accuracy_score#from sklearn.model_selection import GridSearchCVfrom sklearn.feature_extraction.text import CountVectorizerfrom sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import confusion_matrixfrom sklearn.metrics import precision_recall_fscore_support, classification_report,confusion_matrix, precision_recall_curveimport matplotlib.pyplot as pltwith open('Spam_Ham_Config.json') as config_file:  config = json.load(config_file)TrainingData_path = config["TRAINING_DATA_PATH"]Log_path = config['LOG_PATH']vectorizer_filename = config['TEST_VECTORIZER']model_filename = config['TEST_MODEL']nltk_path = config['NLTK_PATH']logging.basicConfig(filename=Log_path, level=logging.INFO, format='%(asctime)s|%(levelname)s:%(lineno)d]%(message)s')#vectorizer_filename = "vectorizer.pk"#model_filename = "spam_ham_model.pk"def text_process(mess):    try:        assert(type(mess) == str)        cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', mess)        cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', cleaned)        cleaned = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr', cleaned)        cleaned = re.sub(r'\d+(\.\d+)?', ' ', cleaned)        cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)        cleaned = re.sub(r'\s+', ' ', cleaned)        cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())        nopunc=''.join(cleaned)        return ' '.join(word.lower() for word in nopunc.split() if word not in stopwords.words('english') if len(word) != 1)    except Exception as e:        print("Exception in loading Text Process", e)        logging.error(f'Exception in loading Text Process:{e}')# This function plots the confusion matrices given y_i, y_i_hat.def app():    #st.title('Check my learning IQ 🐼')    #st.subheader('My Existing Learning set 🐼')    training_df1 = pd.read_csv(TrainingData_path, encoding='iso-8859-1', names=['labels', 'training_messages'], engine='c')    #st.write(training_df1)    #st.write('training data shape = ', training_df1.shape)    #st.write('Spam  and Ham Count in Training Data Set')    training_df1['labels'] = training_df1['labels'].str.upper()    #st.write( training_df1.labels.value_counts())    #st.write('HAM = ', training_df1.HAM.value_counts())    #st.subheader("Add more ideas to enhance my learning 🐼")    #uploaded_file = st.file_uploader("Choose a file")    try:        new_df = pd.read_csv('Flow_TrainingData_bkp.csv', encoding='iso-8859-1', names=['labels', 'training_messages'], engine='c')  # , encoding='unicode_escape')        new_df['labels'] = new_df['labels'].str.upper()        training_df = pd.concat([training_df1, new_df], axis=0)        training_df.dropna(subset = ['training_messages'], inplace = True)        training_df = training_df.dropna()        training_df.reset_index(inplace=True)        training_df = training_df[['labels', 'training_messages']]        training_df['labels'] = training_df['labels'].str.upper()        training_df['training_messages'] = training_df['training_messages'].apply(text_process)        training_df.dropna(subset=['training_messages'], inplace=True)        training_df = training_df.dropna(axis=1)        training_df.reset_index(inplace=True)        training_df.reset_index(inplace=True)        training_df.to_csv('dataaaa.csv')        vectorizer = TfidfVectorizer(ngram_range=(1, 2))        X_ngrams = vectorizer.fit_transform(training_df['training_messages'])    #################################################        x_train, x_test, y_train, y_test = train_test_split(X_ngrams, training_df['labels'], shuffle=True, test_size=0.2,                                                            random_state=42)    except Exception as exception:        print("Exception in Reading Module...",exception)        logging.error(f'Exception in Reading Testing files..{exception}')app()