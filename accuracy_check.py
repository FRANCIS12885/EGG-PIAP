import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

selfDirectory = os.getcwd()



def conf(a, label):
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = a, display_labels = [label, "NOT"])
    cm_display.plot()
    plt.show()



def calculate(v, a, happy_present):
    pain_strain_axis = 2  # oś pionowa rozgraniczająca PAIN i STRAIN
    neutral_axis = 1  # oś pozioma ograniczająca NEUTRAL z góry
    happy_axis_v = 15  # oś pionowa ograniczająca HAPPY z prawej
    happy_axis_a = 1  # oś pozioma ograniczająca HAPPY z góry

    if v < pain_strain_axis and a > neutral_axis:
        return 'PAIN'
    elif v > pain_strain_axis and v < happy_axis_v and a < happy_axis_a and happy_present:
        return 'HAPPY'
    elif v > pain_strain_axis and a > neutral_axis:
        return 'STRAIN'
    else:
        return 'NEUTRAL'

def val_aro_state(main_dic, happy_present):
    folderPath = main_dic + "\\VAL_ARO"
    os.chdir(folderPath)
    all_files = glob.glob('**/*.csv', recursive=True)
    datas = []
    states = ['PAIN', 'STRAIN', 'NEUTRAL']
    if happy_present:
        states.append('HAPPY')
    for file in all_files:

        csv = pd.read_csv(file)
        rows = []
        for row in csv.itertuples():
            my_list = [row.VAL, row.ARO, file.split('_')[2].split('.')[0]]

            rows.append(my_list)
        datas += rows



    return datas


def confusion(datas, happy_present, display):
    y_test = []
    y_pred = []
    a = True
    if happy_present:
        states = ['NEUTRAL', 'HAPPY', 'PAIN', 'STRAIN']
    else:
        states = ['NEUTRAL', 'PAIN', 'STRAIN']
    for state in states:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for data in datas:
            t_state = data[2]
            if not happy_present and t_state == 'HAPPY':
                t_state = 'NEUTRAL'
            c_state = calculate(data[0], data[1],happy_present)
            if state == t_state and state == c_state:
                TP += 1
            elif state != t_state and state != c_state:
                TN += 1
            elif state == t_state and state != c_state:
                FP += 1
            elif state != t_state and state == c_state:
                FN += 1
            if a:
                y_test.append(t_state)
                y_pred.append(c_state)
        a = False
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * (precision * recall) / (precision + recall)
        ACC = (TP + TN) / (TP + TN + FP + FN)

        matrix_type = 0
        if matrix_type:
            conf(np.array([[TP, FN], [FP, TN]]), state)
            print(state)
            print('|',TP,FN,'|')
            print('|', FP, TN, '|')
            print('Precision:', float("{:.4f}".format(precision)))
            print('Recall:', float("{:.4f}".format(recall)))
            print('F1:', float("{:.4f}".format(F1)))
            print('Accuracy: ',float("{:.4f}".format(ACC)) )
            print()
    result = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=result,
                                  display_labels=states)
    result = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(result)
    result1 = classification_report(y_test, y_pred)
    print('Classification Report:', )
    print(result1)
    result2 = accuracy_score(y_test, y_pred)
    print('Accuracy:', result2)

    if display:
        disp.plot()
        disp.ax_.set_title('Val-Aro Model')
        plt.show()



if __name__ == "__main__":
    selfDirectory = os.getcwd()
    col = ['POW.F4.Alpha', 'POW.AF3.Alpha', 'POW.AF4.Alpha', 'POW.F3.BetaL', 'POW.F4.BetaL', 'POW.F3.Gamma',
           'POW.F4.Gamma', 'POW.AF3.Gamma', 'POW.AF4.Gamma']

    ml_data = pd.read_csv(selfDirectory + "\\ML_FQ\\ML_FQ.csv", index_col=False, usecols=col + ['state'])
    confusion(val_aro_state(selfDirectory, False), False)
