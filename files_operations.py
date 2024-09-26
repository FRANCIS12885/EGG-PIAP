import pandas as pd
import os, glob
from sklearn.utils import shuffle



'''
Timestamp,OriginalTimestamp,EEG.Counter,EEG.Interpolated,EEG.AF3,EEG.F7,EEG.F3,EEG.FC5,
EEG.T7,EEG.P7,EEG.O1,EEG.O2,EEG.P8,EEG.T8,EEG.FC6,EEG.F4,EEG.F8,EEG.AF4,EEG.RawCq,
EEG.Battery,EEG.BatteryPercent,MarkerIndex,MarkerType,MarkerValueInt,EEG.MarkerHardware,
CQ.AF3,CQ.F7,CQ.F3,CQ.FC5,CQ.T7,CQ.P7,CQ.O1,CQ.O2,CQ.P8,CQ.T8,CQ.FC6,CQ.F4,CQ.F8,CQ.AF4,
CQ.Overall,EQ.SampleRateQuality,EQ.OVERALL,EQ.AF3,EQ.F7,EQ.F3,EQ.FC5,EQ.T7,EQ.P7,EQ.O1,
EQ.O2,EQ.P8,EQ.T8,EQ.FC6,EQ.F4,EQ.F8,EQ.AF4,POW.AF3.Theta,POW.AF3.Alpha,POW.AF3.BetaL,
POW.AF3.BetaH,POW.AF3.Gamma,POW.F7.Theta,POW.F7.Alpha,POW.F7.BetaL,POW.F7.BetaH,
POW.F7.Gamma,POW.F3.Theta,POW.F3.Alpha,POW.F3.BetaL,POW.F3.BetaH,POW.F3.Gamma,
POW.FC5.Theta,POW.FC5.Alpha,POW.FC5.BetaL,POW.FC5.BetaH,POW.FC5.Gamma,POW.T7.Theta,
POW.T7.Alpha,POW.T7.BetaL,POW.T7.BetaH,POW.T7.Gamma,POW.P7.Theta,POW.P7.Alpha,
POW.P7.BetaL,POW.P7.BetaH,POW.P7.Gamma,POW.O1.Theta,POW.O1.Alpha,POW.O1.BetaL,
POW.O1.BetaH,POW.O1.Gamma,POW.O2.Theta,POW.O2.Alpha,POW.O2.BetaL,POW.O2.BetaH,
POW.O2.Gamma,POW.P8.Theta,POW.P8.Alpha,POW.P8.BetaL,POW.P8.BetaH,POW.P8.Gamma,
POW.T8.Theta,POW.T8.Alpha,POW.T8.BetaL,POW.T8.BetaH,POW.T8.Gamma,POW.FC6.Theta,
POW.FC6.Alpha,POW.FC6.BetaL,POW.FC6.BetaH,POW.FC6.Gamma,POW.F4.Theta,POW.F4.Alpha,
POW.F4.BetaL,POW.F4.BetaH,POW.F4.Gamma,POW.F8.Theta,POW.F8.Alpha,POW.F8.BetaL,POW.F8.BetaH,
POW.F8.Gamma,POW.AF4.Theta,POW.AF4.Alpha,POW.AF4.BetaL,POW.AF4.BetaH,POW.AF4.Gamma
'''


def POW_from_csv(main_dic, happy_present, cols):
    #targetColumns = '''POW.AF3.Theta,POW.AF3.Alpha,POW.AF3.BetaL,POW.AF3.BetaH,POW.AF3.Gamma,POW.F7.Theta,POW.F7.Alpha,POW.F7.BetaL,POW.F7.BetaH,POW.F7.Gamma,POW.F3.Theta,POW.F3.Alpha,POW.F3.BetaL,POW.F3.BetaH,POW.F3.Gamma,POW.FC5.Theta,POW.FC5.Alpha,POW.FC5.BetaL,POW.FC5.BetaH,POW.FC5.Gamma,POW.T7.Theta,POW.T7.Alpha,POW.T7.BetaL,POW.T7.BetaH,POW.T7.Gamma,POW.P7.Theta,POW.P7.Alpha,POW.P7.BetaL,POW.P7.BetaH,POW.P7.Gamma,POW.O1.Theta,POW.O1.Alpha,POW.O1.BetaL,POW.O1.BetaH,POW.O1.Gamma,POW.O2.Theta,POW.O2.Alpha,POW.O2.BetaL,POW.O2.BetaH,POW.O2.Gamma,POW.P8.Theta,POW.P8.Alpha,POW.P8.BetaL,POW.P8.BetaH,POW.P8.Gamma,POW.T8.Theta,POW.T8.Alpha,POW.T8.BetaL,POW.T8.BetaH,POW.T8.Gamma,POW.FC6.Theta,POW.FC6.Alpha,POW.FC6.BetaL,POW.FC6.BetaH,POW.FC6.Gamma,POW.F4.Theta,POW.F4.Alpha,POW.F4.BetaL,POW.F4.BetaH,POW.F4.Gamma,POW.F8.Theta,POW.F8.Alpha,POW.F8.BetaL,POW.F8.BetaH,POW.F8.Gamma,POW.AF4.Theta,POW.AF4.Alpha,POW.AF4.BetaL,POW.AF4.BetaH,POW.AF4.Gamma'''.split(',')
    targetColumns = cols
    #print(targetColumns)

    # Ścieżka do folderu zawierającego pliki CSV
    folderPath = main_dic + "\\Recordings"
    os.chdir(folderPath)
    outputDirectoryML = main_dic + "\\ML_FQ"
    # Pobranie wszystkich plików CSV w folderze
    allFiles = glob.glob('**/*.csv', recursive=True)
    print(allFiles)

    tp = 5
    tk = 25
    Hz = 64

    allCsvFiles = []

    painDataCSV = []
    strainDataCSV = []
    neutralDataCSV = []
    happyDataCSV = []

    totalCSV = []

    fileNames = []
    fileInfos = []
    for file in allFiles:
        fileInfo = file.split('_')
        fileState = fileInfo[0]
        fileName = fileInfo[1]

        if fileName not in fileNames:
            fileNames.append(fileName)
            fileInfos.append([fileName,fileState,[]])
        if fileName in fileNames:
            fileInfos[fileNames.index(fileName)][2].append(file)


    for file in fileInfos:
        csvFiles = []
        for name in file[2]:
            #try:
            csv = pd.read_csv(name, skiprows=1, usecols=targetColumns)
            #except:
            #    print("Could not open:"+name)

            csv = csv[(csv.index >= tp*Hz) & (csv.index <= tk*Hz)]
            csv = csv[~csv.isnull().any(axis=1)]

            csvFiles.append(csv)
        csv = pd.concat(csvFiles)
        csv = csv.reset_index(drop=True)

        match file[1]:
            case 'NEUTRAL':
                neutralDataCSV.append([file[0],csv])
                csv['state'] = 'NEUTRAL'
                totalCSV.append(csv)
            case 'PAIN':
                painDataCSV.append([file[0],csv])
                csv['state'] = 'PAIN'
                totalCSV.append(csv)
            case 'STRAIN':
                strainDataCSV.append([file[0],csv])
                csv['state'] = 'STRAIN'
                totalCSV.append(csv)
            case 'HAPPY':
                happyDataCSV.append([file[0], csv])
                if happy_present:
                    csv['state'] = 'HAPPY'
                else:
                    csv['state'] = 'NEUTRAL'
                totalCSV.append(csv)
                totalCSV = shuffle(totalCSV)
    pd.concat(totalCSV).to_csv(outputDirectoryML + "\\ML_FQ.csv", index=False)
    print('1: Successfully extracted data from raw CSV files')
    return [neutralDataCSV,painDataCSV,strainDataCSV, happyDataCSV]




#print(POW_from_csv()[0][0][1].to_string())