import pickle
import json
import logging
import scipy
with open('A2P_P2P_Config.json') as config_file:
    config = json.load(config_file)

vectorizer_save = config['VECTORIZER']
model_save = config['MODEL']
logpath = config['LOG_PATH']
P2P_File_enable = config['PROCESS_FILE_EXTENSION']

vectorizer = pickle.load(open(vectorizer_save, 'rb'))
A2P_P2P_detect_model = pickle.load(open(model_save, 'rb'))

#LogCreationPath = Log_tag + Log_tag
logging.basicConfig(filename=logpath,level=logging.INFO,format='%(asctime)s|%(levelname)s:%(lineno)d]%(message)s')

def A2P_P2P_predict(text):
    global vectorizer
    global A2P_P2P_detect_model
    try:

        X = vectorizer.transform([text])
        sparse_pred_mat = scipy.sparse.csr_matrix(X)
        pred = A2P_P2P_detect_model.predict(sparse_pred_mat)[0]
        #logging.info('Predict successfull....')
        return pred

    except Exception as exception:
        print("Exception in process_Predict:", exception)
        logging.error(f'Exception in process_Predict:{exception}')
if __name__ == '__main__':
    text = 'mascomchat let check account balance buy airtime data bundles whatsapp get started save number phonenumbr send hi|Unnamed: 1'
    res = A2P_P2P_predict(text)
    #pred = A2P_P2P_detect_model.predict(vectorizer.transform(["i have got a new phone. its from Apple.. and i love it!"]))[0]
    print("predicted class:", res)
    #print(pred)