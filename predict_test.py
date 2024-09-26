import pickle, os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def mlp_predict(a, model):
    pre = np.argmax(model.predict(a), axis=1)
    dec_pre = label_encoder.inverse_transform(pre)
    return(dec_pre)

selfDirectory = os.getcwd()
picklesDirectory = selfDirectory + '\\pickles'

os.chdir(picklesDirectory)

test_data = [[np.random.randint(0,20,9)] for i in range(10)]


with open(picklesDirectory +"\\KNN_3.pkl", 'rb') as file:
    knn_model = pickle.load(file)
with open(picklesDirectory +"\\SVM_POLY.pkl", 'rb') as file:
    svm_model = pickle.load(file)
with open(picklesDirectory +"\\RF.pkl", 'rb') as file:
    rf_model = pickle.load(file)
with open(picklesDirectory +"\\MLP.h5", 'rb') as file:
    mlp_model = load_model('MLP.h5')


models = [knn_model, svm_model, rf_model, mlp_model]

for model in models:
    print('predicting using', model)
    a = model.predict(test_data[0])
    print(a)
