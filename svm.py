from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os, pickle, joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


selfDirectory = os.getcwd()
picklesDirectory = os.path.abspath(os.path.dirname(__file__)) + "\\pickles"


def svm_fq(data, target_col, cols, export, happy_present, ts, display):
    if happy_present:
        states = ['HAPPY','NEUTRAL','PAIN','STRAIN']
    else:
        states = ['NEUTRAL','PAIN','STRAIN']
    df = data

    factor = pd.factorize(df[target_col])
    df[target_col] = factor[0]
    definitions = factor[1]

    y_data = df[target_col]
    data_pass = df.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data_pass, y_data, test_size=ts, random_state=0, stratify=y_data)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)


    poly = svm.SVC(kernel='poly', C=1, degree=3).fit(X_train, y_train)

    poly_pred = poly.predict(X_test)

    reversefactor = dict(zip(range(len(states)), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    poly_pred = np.vectorize(reversefactor.get)(poly_pred)

    ratio_label = str(int((1-ts)*100))+'/'+str(int(ts*100))
    cm = confusion_matrix(y_test, poly_pred)

    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=states)
        disp.plot()
        disp.ax_.set_title('SVM - test set '+ratio_label)
        plt.show()



    print("POLY: ")
    result = confusion_matrix(y_test, poly_pred)
    print('Confusion Matrix:')
    print(result)
    result1 = classification_report(y_test, poly_pred)
    print('Classification Report:',)
    print (result1)
    result2 = accuracy_score(y_test, poly_pred)
    print('Accuracy:',result2)
    print()

    if export:
        os.chdir(picklesDirectory)
        polyPickle = open('SVM.pkl', 'wb')
        pickle.dump(poly, polyPickle)
        polyPickle.close()
        joblib.dump(scaler, "SVM_scaler.pkl")

if __name__=="__main__":
    col = ['POW.F4.Alpha', 'POW.AF3.Alpha', 'POW.AF4.Alpha', 'POW.F3.BetaL', 'POW.F4.BetaL', 'POW.F3.Gamma',
           'POW.F4.Gamma', 'POW.AF3.Gamma', 'POW.AF4.Gamma']
    svm_data = pd.read_csv(selfDirectory + "\\ML_FQ\\ML_FQ.csv", index_col=False, usecols=col+['state'])
    svm_fq(svm_data, 'state', col)
