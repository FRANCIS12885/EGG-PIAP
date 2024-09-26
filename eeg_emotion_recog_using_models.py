"""Example program to show how to read a multi-channel time series from LSL."""
from pylsl import StreamInlet, resolve_stream
from math import sqrt
import pickle, os, time, joblib
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter

selfDirectory = os.getcwd()

label_encoder = None



def calculate_state(v, a):
    pain_strain_axis = 2  # oś pionowa rozgraniczająca PAIN i STRAIN
    neutral_axis = 1  # oś pozioma ograniczająca NEUTRAL z góry
    happy_axis_v = 15  # oś pionowa ograniczająca HAPPY z prawej
    happy_axis_a = 1  # oś pozioma ograniczająca HAPPY z góry
    happy_present = False
    if v < pain_strain_axis and a > neutral_axis:
        return 'PAIN'
    elif v > pain_strain_axis and v < happy_axis_v and a < happy_axis_a and happy_present:
        return 'HAPPY'
    elif v > pain_strain_axis and a > neutral_axis:
        return 'STRAIN'
    else:
        return 'NEUTRAL'

def d(x,y):
    return sqrt(x**2+y**2)

def get_eeg_power_data(sample_list):
    current_sample = np.array(sample_list[1:71])
    '''
    [
    "AF3/theta","AF3/alpha","AF3/betaL","AF3/betaH","AF3/gamma",
    "F7/theta","F7/alpha","F7/betaL","F7/betaH","F7/gamma",
    "F3/theta","F3/alpha","F3/betaL","F3/betaH","F3/gamma",
    "FC5/theta","FC5/alpha","FC5/betaL","FC5/betaH","FC5/gamma",
    "T7/theta","T7/alpha","T7/betaL","T7/betaH","T7/gamma",
    "P7/theta","P7/alpha","P7/betaL","P7/betaH","P7/gamma",
    "O1/theta","O1/alpha","O1/betaL","O1/betaH","O1/gamma",
    "O2/theta","O2/alpha","O2/betaL","O2/betaH","O2/gamma",
    "P8/theta","P8/alpha","P8/betaL","P8/betaH","P8/gamma",
    "T8/theta","T8/alpha","T8/betaL","T8/betaH","T8/gamma",
    "FC6/theta","FC6/alpha","FC6/betaL","FC6/betaH","FC6/gamma",
    "F4/theta","F4/alpha","F4/betaL","F4/betaH","F4/gamma",
    "F8/theta","F8/alpha","F8/betaL","F8/betaH","F8/gamma",
    "AF4/theta","AF4/alpha","AF4/betaL","AF4/betaH","AF4/gamma"
    ]
    '''


    bF4 = current_sample[57]
    bF3 = current_sample[12]
    aAF3 = current_sample[1]
    aAF4 = current_sample[66]
    aF4 = current_sample[56]

    gAF3 = current_sample[4]
    gAF4 = current_sample[69]
    gF3 = current_sample[14]
    gF4 = current_sample[59]

    valence = - (bF4 / bF3 * d(bF4, bF3) + gAF3 / gAF4 * d(gF4, gF3)) + aAF4 / aAF3 * d(aAF4, aAF3) + gF4 * 2
    arousal = gF4 * 2


    val_aro_state = calculate_state(valence, arousal)

    #return [aF4, aAF3, aAF4, bF3, bF4, gF3, gF4, gAF3, gAF4]
    return [aAF3, gAF3, bF3, gF3, aF4, bF4, gF4, aAF4, gAF4], val_aro_state


def mlp_predict(a, model):
    pre = np.argmax(model.predict(a, verbose=0), axis=1)
    return(pre)

def main():
    global label_encoder
    print("Looking for a stream...")
    # first resolve a Motion stream on the lab network
    streams = resolve_stream('type', 'Band-Power') # You can try other stream types such as: EEG, EEG-Quality, Contact-Quality, Performance-Metrics, Band-Power
    print(streams)
    print("Available streams are printed above.")
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    print("Inlet for the stream has been created.")
    print("Starting broadcasting EEG signals...")
    printing_counter = 0
    picklesDirectory = selfDirectory + '\\pickles'

    os.chdir(picklesDirectory)
    with open(picklesDirectory + "\\KNN_3.pkl", 'rb') as file:
        knn_model = pickle.load(file)
    with open(picklesDirectory + "\\SVM.pkl", 'rb') as file:
        svm_model = pickle.load(file)
    with open(picklesDirectory + "\\RF.pkl", 'rb') as file:
        rf_model = pickle.load(file)
    with open(picklesDirectory + "\\MLP.h5", 'rb') as file:
        mlp_model = load_model('MLP.h5')
    knn_scaler = joblib.load('KNN_scaler.pkl')
    svm_scaler = joblib.load('SVM_scaler.pkl')
    rf_scaler = joblib.load('RF_scaler.pkl')
    mlp_scaler = joblib.load('MLP_scaler.pkl')

    models = [knn_model, svm_model, rf_model]
    scalers = [knn_scaler, svm_scaler, rf_scaler]

    model_nr = 2

    happy_present = False

    label_encoder = LabelEncoder()
    if happy_present:
        states = ['HAPPY', 'NEUTRAL', 'PAIN', 'STRAIN']
    else:
        states = ['NEUTRAL', 'PAIN', 'STRAIN']
    label_encoder.fit(states)

    definitions = pd.Index(states, dtype='object')

    reversefactor = dict(zip(range(len(states)), definitions))

    col = ['POW.AF3.Alpha', 'POW.AF3.Gamma', 'POW.F3.BetaL', 'POW.F3.Gamma',
           'POW.F4.Alpha', 'POW.F4.BetaL', 'POW.F4.Gamma', 'POW.AF4.Alpha', 'POW.AF4.Gamma']
    print_counter = 0
    while True:
        # Returns a tuple (sample,timestamp) where sample is a list of channel values and timestamp is the capture time of the sample on the remote machine,
        # or (None,None) if no new sample was available
        sample, timestamp = inlet.pull_sample()
        if timestamp != None:
            #print(timestamp, sample)
            band_stream, val_aro_state = get_eeg_power_data(sample)
            data = np.array(band_stream).reshape(1, -1)
            '''
            if model_nr == 3:
                pred = mlp_predict(mlp_scaler.transform(data), mlp_model)
             else:
                pred = models[model_nr].predict(scalers[model_nr].transform(data))
            state = np.vectorize(reversefactor.get)(pred)[0]
            '''
            knn_pred = knn_model.predict(knn_scaler.transform(data))
            svm_pred = svm_model.predict(svm_scaler.transform(data))
            rf_pred = rf_model.predict(rf_scaler.transform(data))
            mlp_pred = mlp_predict(mlp_scaler.transform(data), mlp_model)

            knn_pred = np.vectorize(reversefactor.get)(knn_pred)[0]
            svm_pred = np.vectorize(reversefactor.get)(svm_pred)[0]
            rf_pred = np.vectorize(reversefactor.get)(rf_pred)[0]
            mlp_pred = np.vectorize(reversefactor.get)(mlp_pred)[0]
            states = [val_aro_state, knn_pred, svm_pred, rf_pred, mlp_pred]
            counter = Counter(states)
            most_common_elements = counter.most_common()

            # Find the highest count
            highest_count = most_common_elements[0][1]

            # Find all elements with the highest count
            most_common_items = [item for item, count in most_common_elements if count == highest_count]
        print(print_counter,'VAL_ARO:',states[0], '| \t KNN:', states[1],'| \t SVM:', states[2],'| \t RF:', states[3],'| \t MLP:', states[4],'| \t MOST_COMMON:', most_common_items)
        time.sleep(0.5)
        print_counter += 1
        if print_counter >= 10:
            print_counter=0


if __name__ == "__main__":
    main()
