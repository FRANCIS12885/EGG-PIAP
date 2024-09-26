from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import joblib


selfDirectory = os.getcwd()
picklesDirectory = os.path.abspath(os.path.dirname(__file__)) + "\\pickles"


def random_forest_fq(data, target_col, cols, export, happy_present, ts, display):
    if  happy_present:
        states = ['HAPPY','NEUTRAL','PAIN','STRAIN']
    else:
        states = ['NEUTRAL', 'PAIN', 'STRAIN']

    df = data
    factor = pd.factorize(df[target_col])
    df[target_col] = factor[0]
    definitions = factor[1]

    y_data = df[target_col]
    data_pass = df.drop(target_col, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data_pass, y_data, test_size=ts, random_state = 0, stratify=y_data)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)

    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    reversefactor = dict(zip(range(len(states)), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)

    ratio_label = str(int((1-ts)*100))+'/'+str(int(ts*100))
    cm = confusion_matrix(y_test, y_pred)

    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels= states)
        disp.plot()
        disp.ax_.set_title('Random Forest - test set '+ratio_label)
        plt.show()

    result = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(result)
    result1 = classification_report(y_test, y_pred)
    print('Classification Report:',)
    print (result1)
    result2 = accuracy_score(y_test, y_pred)
    print('Accuracy:',result2)
    print()
    if export:
        os.chdir(picklesDirectory)
        rfPickle = open('RF.pkl', 'wb')
        pickle.dump(classifier, rfPickle)
        rfPickle.close()
        joblib.dump(scaler, "RF_scaler.pkl")


if __name__=="__main__":
    col = ['POW.F4.Alpha', 'POW.AF3.Alpha', 'POW.AF4.Alpha', 'POW.F3.BetaL', 'POW.F4.BetaL', 'POW.F3.Gamma',
           'POW.F4.Gamma', 'POW.AF3.Gamma', 'POW.AF4.Gamma']
    ml_data = pd.read_csv(selfDirectory + "\\ML_FQ\\ML_FQ.csv", index_col=False, usecols=col+['state'])
    random_forest_fq(ml_data, 'state', col)
