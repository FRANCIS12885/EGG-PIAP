import pandas as pd
import numpy as np
from math import sqrt


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

def d(x,y):
    return sqrt(x**2+y**2)

def calculateARO_VAL(data):

    betaSide = 'L'

    aF4 = data['POW.F4.Alpha']
    aAF3 = data['POW.AF3.Alpha']
    aAF4 = data['POW.AF4.Alpha']

    bF3 = data['POW.F3.Beta' + betaSide]
    bF4 = data['POW.F4.Beta' + betaSide]

    gF3 = data['POW.F3.Gamma']
    gF4 = data['POW.F4.Gamma']
    gAF3 = data['POW.AF3.Gamma']
    gAF4 = data['POW.AF4.Gamma']


    va = {'VAL': [], 'ARO': []}
    for i in range(len(aF4)):

        '''
        valence = aF4/bF4 - aF3/bF3
        valence = np.mean([aF4-tAF4 , aF3-tAF3])
        arousal_scaled = arousal - 2.5
        '''

        sensorNames = [
        'F3', 'F4', 'AF4', 'AF4','F7',
        'F8', 'FC5', 'FC6', 'T7', 'T8',
        'P7', 'P8', 'O1', 'O2'
        ]
        val_scale = 0
        aro_scale = 0

        #val = (aF4[i] / bF4[i] - aF3[i] / bF3[i])
        #val = np.mean([aF4[i]-tAF4[i] , aF3[i]-tAF3[i]])
        #aro = np.mean([bAF3[i] / aAF3[i], bAF4[i] / aAF4[i], bF3[i] / aF3[i], bF4[i] / aF4[i]])

        val = - (bF4[i]/bF3[i]*d(bF4[i],bF3[i]) + gF3[i]/gF4[i]*d(gF4[i],gF3[i])) + aAF4[i]/aAF3[i]*d(aAF4[i],aAF3[i])+gF4[i]*2
        aro = gF4[i]*2

        #val = data['POW.AF3.Gamma'][i]
        #aro = data['POW.AF4.Gamma'][i]

        va['VAL'].append(val + val_scale)
        va['ARO'].append(aro + aro_scale)

    return pd.DataFrame(va)

def val_aro_to_csv(data_stream, main_dic, happy_present):

    statesData = data_stream
    #print(statesData)

    neutralCSV = data_stream[0]
    painCSV = data_stream[1]
    strainCSV = data_stream[2]
    happyCSV = data_stream[3]

    painDataARO_VAL= []
    strainDataARO_VAL = []
    neutralDataARO_VAL = []
    happyDataARO_VAL = []

    selfDirectory = main_dic
    outputDirectory = selfDirectory + "\\VAL_ARO"
    outputDirectoryKNN = selfDirectory + "\\KNN_VAL_ARO"
    total_val_aro = []


    for data in neutralCSV:
        neutralDataARO_VAL.append([data[0], calculateARO_VAL(data[1])])
    for data in painCSV:
        painDataARO_VAL.append([data[0], calculateARO_VAL(data[1])])
    for data in strainCSV:
        strainDataARO_VAL.append([data[0], calculateARO_VAL(data[1])])
    for data in happyCSV:
        happyDataARO_VAL.append([data[0], calculateARO_VAL(data[1])])
    print('2: Successfully calculated VAL and ARO values')
    for valAroDataFrame in neutralDataARO_VAL:
        outputFileName = valAroDataFrame[0] + "_TARGET_NEUTRAL.csv"
        valAroDataFrame[1].to_csv(outputDirectory + "\\" + outputFileName)
        #valAroDataFrame[1]['state'] = 2
        valAroDataFrame[1]['state'] = 'NEUTRAL'
        total_val_aro.append(valAroDataFrame[1])
    for valAroDataFrame in painDataARO_VAL:
        outputFileName = valAroDataFrame[0] + "_TARGET_PAIN.csv"
        valAroDataFrame[1].to_csv(outputDirectory + "\\" + outputFileName)
        #valAroDataFrame[1]['state'] = 0
        valAroDataFrame[1]['state'] = 'PAIN'
        total_val_aro.append(valAroDataFrame[1])
    for valAroDataFrame in strainDataARO_VAL:
        outputFileName = valAroDataFrame[0] + "_TARGET_STRAIN.csv"
        valAroDataFrame[1].to_csv(outputDirectory + "\\" + outputFileName)
        #valAroDataFrame[1]['state'] = 1
        valAroDataFrame[1]['state'] = 'STRAIN'
        total_val_aro.append(valAroDataFrame[1])
    for valAroDataFrame in happyDataARO_VAL:
        a = 'HAPPY'
        if not happy_present:
            a = 'NEUTRAL'
        outputFileName = valAroDataFrame[0] + "_TARGET_"+a+".csv"
        valAroDataFrame[1].to_csv(outputDirectory + "\\" + outputFileName)
        #valAroDataFrame[1]['state'] = 3
        valAroDataFrame[1]['state'] = a
        total_val_aro.append(valAroDataFrame[1])

    print("3: Successfully saved VAl and ARO as CSV files")

    pd.concat(total_val_aro).to_csv(outputDirectoryKNN + "\\KNN_VAL_ARO.csv", index=False)

'''
if __name__ == "__main__":
    val_aro_to_csv(fo.POW_from_csv())
'''