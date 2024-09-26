from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import os, pickle
import numpy as np
import joblib


selfDirectory = os.getcwd()
picklesDirectory = os.path.abspath(os.path.dirname(__file__)) + "\\pickles"


def knn_fq(data, target_col, cols, k, export, happy_present, ts, display):
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
    a_test = X_test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.transform(X_test.values)

    knn_clf=KNeighborsClassifier(n_neighbors=k)

    knn_clf.fit(X_train, y_train)
    y_pred=knn_clf.predict(X_test) #These are the predicted output values
    reversefactor = dict(zip(range(len(states)), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)


    result = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(result)
    result1 = classification_report(y_test, y_pred)
    print('Classification Report:',)
    print (result1)
    result2 = accuracy_score(y_test,y_pred)
    print('Accuracy:',result2)
    if export:
        os.chdir(picklesDirectory)
        knnPickle = open('KNN_'+str(k)+'.pkl', 'wb')
        pickle.dump(knn_clf, knnPickle)
        knnPickle.close()
        joblib.dump(scaler, "KNN_scaler.pkl")

    cm = confusion_matrix(y_test, y_pred)

    #print('KNN format:')
    #print('Test values:',y_test)
    #print('Predicted values:',y_pred)
    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=states)

        ratio_label = str(int((1-ts)*100))+'/'+str(int(ts*100))
        disp.plot()
        disp.ax_.set_title('KNN - test set '+ratio_label)
        plt.show()
    #print('X_test returned by knn:',X_test)
    return a_test, y_test, reversefactor


