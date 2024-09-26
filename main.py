import pandas as pd
import files_operations as fo
import val_aro_calculator as vac
import plot_val_aro as pva
import os, pickle, joblib, sys
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

all_pows = ['POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL','POW.AF3.BetaH', 'POW.AF3.Gamma','POW.F7.Theta', 'POW.F7.Alpha','POW.F7.BetaL', 'POW.F7.BetaH', 'POW.F7.Gamma','POW.F3.Theta','POW.F3.Alpha', 'POW.F3.BetaL', 'POW.F3.BetaH', 'POW.F3.Gamma','POW.FC5.Theta', 'POW.FC5.Alpha', 'POW.FC5.BetaL', 'POW.FC5.BetaH','POW.FC5.Gamma','POW.T7.Theta', 'POW.T7.Alpha', 'POW.T7.BetaL','POW.T7.BetaH', 'POW.T7.Gamma','POW.P7.Theta', 'POW.P7.Alpha','POW.P7.BetaL', 'POW.P7.BetaH', 'POW.P7.Gamma','POW.O1.Theta','POW.O1.Alpha', 'POW.O1.BetaL', 'POW.O1.BetaH', 'POW.O1.Gamma','POW.O2.Theta', 'POW.O2.Alpha', 'POW.O2.BetaL', 'POW.O2.BetaH','POW.O2.Gamma','POW.P8.Theta', 'POW.P8.Alpha', 'POW.P8.BetaL','POW.P8.BetaH', 'POW.P8.Gamma','POW.T8.Theta', 'POW.T8.Alpha','POW.T8.BetaL', 'POW.T8.BetaH', 'POW.T8.Gamma','POW.FC6.Theta','POW.FC6.Alpha', 'POW.FC6.BetaL', 'POW.FC6.BetaH', 'POW.FC6.Gamma','POW.F4.Theta', 'POW.F4.Alpha', 'POW.F4.BetaL', 'POW.F4.BetaH','POW.F4.Gamma','POW.F8.Theta', 'POW.F8.Alpha', 'POW.F8.BetaL','POW.F8.BetaH', 'POW.F8.Gamma','POW.AF4.Theta', 'POW.AF4.Alpha','POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma']


'''
'Timestamp',
'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL','POW.AF3.BetaH', 'POW.AF3.Gamma',
'POW.F7.Theta', 'POW.F7.Alpha','POW.F7.BetaL', 'POW.F7.BetaH', 'POW.F7.Gamma',
'POW.F3.Theta','POW.F3.Alpha', 'POW.F3.BetaL', 'POW.F3.BetaH', 'POW.F3.Gamma',
'POW.FC5.Theta', 'POW.FC5.Alpha', 'POW.FC5.BetaL', 'POW.FC5.BetaH','POW.FC5.Gamma',
'POW.T7.Theta', 'POW.T7.Alpha', 'POW.T7.BetaL','POW.T7.BetaH', 'POW.T7.Gamma',
'POW.P7.Theta', 'POW.P7.Alpha','POW.P7.BetaL', 'POW.P7.BetaH', 'POW.P7.Gamma',
'POW.O1.Theta','POW.O1.Alpha', 'POW.O1.BetaL', 'POW.O1.BetaH', 'POW.O1.Gamma',
'POW.O2.Theta', 'POW.O2.Alpha', 'POW.O2.BetaL', 'POW.O2.BetaH','POW.O2.Gamma',
'POW.P8.Theta', 'POW.P8.Alpha', 'POW.P8.BetaL','POW.P8.BetaH', 'POW.P8.Gamma',
'POW.T8.Theta', 'POW.T8.Alpha','POW.T8.BetaL', 'POW.T8.BetaH', 'POW.T8.Gamma',
'POW.FC6.Theta','POW.FC6.Alpha', 'POW.FC6.BetaL', 'POW.FC6.BetaH', 'POW.FC6.Gamma',
'POW.F4.Theta', 'POW.F4.Alpha', 'POW.F4.BetaL', 'POW.F4.BetaH','POW.F4.Gamma',
'POW.F8.Theta', 'POW.F8.Alpha', 'POW.F8.BetaL','POW.F8.BetaH', 'POW.F8.Gamma',
'POW.AF4.Theta', 'POW.AF4.Alpha','POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma'
'''

col = ['POW.F4.Alpha','POW.AF3.Alpha','POW.AF4.Alpha','POW.F3.BetaL','POW.F4.BetaL','POW.F3.Gamma','POW.F4.Gamma','POW.AF3.Gamma','POW.AF4.Gamma']
#col = all_pows
selfDirectory = os.getcwd()

status = 0

if_calc_val_aro = 1
if_accuracy_check = 1

if_knn = 1
if_svm = 1
if_rf = 1
if_mlp = 1

knn_coef = [3]
export = 1

test_size = 0.2

display = 0

def main():
    data_stream = fo.POW_from_csv(selfDirectory, happy_present=status, cols=col)
    if if_calc_val_aro:
        vac.val_aro_to_csv(data_stream, selfDirectory, happy_present=status)
        # pva.visualize_va(selfDirectory, drawJoined=True)
    print()
    if if_accuracy_check:
        print("--------------------Val-Aro--------------------")
        import accuracy_check as ac
        ac.confusion(ac.val_aro_state(selfDirectory, status), status, display=display)
    print()
    print('----------------------------------------------------------------------------------------------')
    print('-Test-----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------')
    print()
    if if_knn:
        import knn
        print("--------------------KNN--------------------")
        for k in knn_coef:
            print('Data for k =', k)
            ml_data = pd.read_csv(selfDirectory + "\\ML_FQ\\ML_FQ.csv", index_col=False, usecols=col + ['state'])
            knn.knn_fq(ml_data, 'state', col, k, export=export, happy_present=status, ts=test_size, display=display)
            print()
    print()
    if if_svm:
        import svm
        print("--------------------SVM----------------------")
        ml_data = pd.read_csv(selfDirectory + "\\ML_FQ\\ML_FQ.csv", index_col=False, usecols=col + ['state'])
        svm.svm_fq(ml_data, 'state', col, export=export, happy_present=status, ts=test_size, display=display)
    print()
    if if_rf:
        import random_forest as rf
        print("-----------------Random Forest----------------")
        ml_data = pd.read_csv(selfDirectory + "\\ML_FQ\\ML_FQ.csv", index_col=False, usecols=col + ['state'])
        rf.random_forest_fq(ml_data, 'state', col, export=export, happy_present=status, ts=test_size, display=display)
    if if_mlp:
        import mlp
        print("----------------------MLP--------------------")
        ml_data = pd.read_csv(selfDirectory + "\\ML_FQ\\ML_FQ.csv", index_col=False, usecols=col + ['state'])
        mlp.mlp_fq(ml_data, 'state', col, export=export, happy_present=status, ts=test_size, display=display)

    print()
    print('----------------------------------------------------------------------------------------------')
    print('-Evaluation-----------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------')
    print()
    #import test_predict as tp
    #tp.main()

    '''
    picklesDirectory = selfDirectory + '\\pickles'

    os.chdir(picklesDirectory)
    with open(picklesDirectory + "\\KNN_3.pkl", 'rb') as file:
        knn_model = pickle.load(file)

    knn_scaler = joblib.load('KNN_scaler.pkl')
    X_test = knn_scaler.transform(X_test)
    print('X_test received by main:', X_test)
    y_pred = knn_model.predict(X_test)

    print('MAIN format before reversefactor:')
    print('Test values:', y_test)
    print('Predicted values:', y_pred)

    #y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)

    print('MAIN format after reverse factor:')
    print('Test values:', y_test)
    print('Predicted values:', y_pred)

    states = ['NEUTRAL','PAIN','STRAIN']
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=states)

    disp.plot()
    disp.ax_.set_title('KNN Check')
    plt.show()
    '''


if __name__ == "__main__":
    main()

