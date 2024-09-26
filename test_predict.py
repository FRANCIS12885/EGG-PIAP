import pandas as pd
import os, glob, pickle, joblib
import numpy as np
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt




all_pows = ['POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL','POW.AF3.BetaH', 'POW.AF3.Gamma','POW.F7.Theta', 'POW.F7.Alpha','POW.F7.BetaL', 'POW.F7.BetaH', 'POW.F7.Gamma','POW.F3.Theta','POW.F3.Alpha', 'POW.F3.BetaL', 'POW.F3.BetaH', 'POW.F3.Gamma','POW.FC5.Theta', 'POW.FC5.Alpha', 'POW.FC5.BetaL', 'POW.FC5.BetaH','POW.FC5.Gamma','POW.T7.Theta', 'POW.T7.Alpha', 'POW.T7.BetaL','POW.T7.BetaH', 'POW.T7.Gamma','POW.P7.Theta', 'POW.P7.Alpha','POW.P7.BetaL', 'POW.P7.BetaH', 'POW.P7.Gamma','POW.O1.Theta','POW.O1.Alpha', 'POW.O1.BetaL', 'POW.O1.BetaH', 'POW.O1.Gamma','POW.O2.Theta', 'POW.O2.Alpha', 'POW.O2.BetaL', 'POW.O2.BetaH','POW.O2.Gamma','POW.P8.Theta', 'POW.P8.Alpha', 'POW.P8.BetaL','POW.P8.BetaH', 'POW.P8.Gamma','POW.T8.Theta', 'POW.T8.Alpha','POW.T8.BetaL', 'POW.T8.BetaH', 'POW.T8.Gamma','POW.FC6.Theta','POW.FC6.Alpha', 'POW.FC6.BetaL', 'POW.FC6.BetaH', 'POW.FC6.Gamma','POW.F4.Theta', 'POW.F4.Alpha', 'POW.F4.BetaL', 'POW.F4.BetaH','POW.F4.Gamma','POW.F8.Theta', 'POW.F8.Alpha', 'POW.F8.BetaL','POW.F8.BetaH', 'POW.F8.Gamma','POW.AF4.Theta', 'POW.AF4.Alpha','POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma']
col = ['POW.F4.Alpha','POW.AF3.Alpha','POW.AF4.Alpha','POW.F3.BetaL','POW.F4.BetaL','POW.F3.Gamma','POW.F4.Gamma','POW.AF3.Gamma','POW.AF4.Gamma']
#col = all_pows

def mlp_predict(a, model):
    pre = np.argmax(model.predict(a), axis=1)
    return(pre)


def main():
    tp = 5
    tk = 25
    Hz = 64


    os.chdir("C:\\Users\\Franek\\Documents\\Praktyki\\EEG kod\\test_recordings\\")
    allFiles = glob.glob('**/*.csv', recursive=True)
    all_csv = []
    all_state_csv = []

    print(allFiles)
    for file in allFiles:
        target_state = file.split('_')[0]
        csv = pd.read_csv(file,  skiprows=1 , usecols = col)

        csv = csv[(csv.index >= tp * Hz) & (csv.index <= tk * Hz)]
        csv = csv[~csv.isnull().any(axis=1)]

        state_csv = [target_state for i in range(len(csv))]
        all_csv.append(csv)
        all_state_csv += state_csv

    states = ['NEUTRAL', 'PAIN', 'STRAIN']
    definitions = pd.Index(states, dtype='object')


    reversefactor = dict(zip(range(len(states)), definitions))



    a = pd.concat(all_csv)

    all_csv = np.array(pd.concat(all_csv))
    #all_state_csv = np.array(pd.concat(all_state_csv))
    #print(a.to_string())
    #print(all_state_csv)

    os.chdir("C:\\Users\\Franek\\Documents\\Praktyki\\EEG kod\\pickles\\")

    with open("KNN_3.pkl", 'rb') as file:
        knn_model = pickle.load(file)
    with open("SVM.pkl", 'rb') as file:
        svm_model = pickle.load(file)
    with open("RF.pkl", 'rb') as file:
        rf_model = pickle.load(file)
    with open("MLP.h5", 'rb') as file:
        mlp_model = load_model('MLP.h5')

    knn_scaler = joblib.load('KNN_scaler.pkl')
    svm_scaler = joblib.load('SVM_scaler.pkl')
    rf_scaler =  joblib.load('RF_scaler.pkl')
    mlp_scaler =  joblib.load('MLP_scaler.pkl')

    knn_pred = knn_model.predict(knn_scaler.transform(all_csv))
    svm_pred = svm_model.predict(svm_scaler.transform(all_csv))
    rf_pred  = rf_model.predict(rf_scaler.transform(all_csv))
    mlp_pred = mlp_predict(mlp_scaler.transform(all_csv), mlp_model)

    knn_pred = np.vectorize(reversefactor.get)(knn_pred)
    svm_pred = np.vectorize(reversefactor.get)(svm_pred)
    rf_pred  = np.vectorize(reversefactor.get)(rf_pred)
    mlp_pred  = np.vectorize(reversefactor.get)(mlp_pred)
    #print(knn_pred)


    for pred, name in zip([knn_pred, svm_pred, rf_pred, mlp_pred], ['KNN','SVM','RF','MLP']):
        print('-----------'+name+'-------------------')
        cm = confusion_matrix(all_state_csv, pred)
        
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=states)
        #disp.plot()
        #disp.ax_.set_title(name+' - evaluation set')
        #plt.show()

        result = cm
        print('Confusion Matrix:')
        print(result)
        result1 = classification_report(all_state_csv, pred)
        print('Classification Report:', )
        print(result1)
        result2 = accuracy_score(all_state_csv, pred)
        print('Accuracy:', result2)







    '''
    tot =  len(knn_pred)
    target = allFiles[0].split('_')[0]
    
    print('KNN:')
    
    for state in states:
        if state==target:
            print(state+':',list(knn_pred).count(state),list(knn_pred).count(state)/tot )
        else:
            print(state + ':', list(knn_pred).count(state))
    
    print()
    print('SVM:')
    for state in states:
        if state == target:
            print(state + ':', list(svm_pred).count(state), list(svm_pred).count(state) / tot)
        else:
            print(state + ':', list(svm_pred).count(state))
    print()
    print('RF:')
    for state in states:
        if state == target:
            print(state + ':', list(rf_pred).count(state), list(rf_pred).count(state) / tot)
        else:
            print(state + ':', list(rf_pred).count(state))
    print()
    print('MLP:')
    for state in states:
        if state == target:
            print(state + ':', list(mlp_pred).count(state), list(mlp_pred).count(state) / tot)
        else:
            print(state + ':', list(mlp_pred).count(state))
    '''

if __name__ == "__main__":
    main()
