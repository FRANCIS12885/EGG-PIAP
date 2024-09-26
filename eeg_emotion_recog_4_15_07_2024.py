"""Example program to show how to read a multi-channel time series from LSL."""
from pylsl import StreamInlet, resolve_stream
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# counter to perform print statement with emotional state
N = 4


'''
good 1
    if v < -1 and a > 2.5:
        return 'PAIN'
    elif v > 0 and a > 2.5:
        return 'STRAIN'
    else:
        return 'NEUTRAL'
        
good 2
    if v < -10 and a > 1.5:
        return 'PAIN'
    elif v > 0 and a > 1.5:
        return 'STRAIN'
    else:
        return 'NEUTRAL'


'''


def calculate_state(v, a):
    pain_strain_axis = 2  # oś pionowa rozgraniczająca PAIN i STRAIN
    neutral_axis = 1  # oś pozioma ograniczająca NEUTRAL z góry
    happy_axis_v = 15  # oś pionowa ograniczająca HAPPY z prawej
    happy_axis_a = 1  # oś pozioma ograniczająca HAPPY z góry

    if v < pain_strain_axis and a > neutral_axis:
        return 'PAIN'
    elif v > pain_strain_axis and v < happy_axis_v and a < happy_axis_a:
        return 'HAPPY'
    elif v > pain_strain_axis and a > neutral_axis:
        return 'STRAIN'
    else:
        return 'NEUTRAL'

def visualize_single(v,a, state):
    df = pd.DataFrame([v,a, state], columns=['VAL', 'ARO', 'STATE'])

    color_map = {'NEUTRAL': 'yellow', 'HAPPY': 'green', 'PAIN': 'red', 'STRAIN': 'blue'}
    colors = df['STATE'].map(color_map)

    plt.figure(figsize=(10, 6))
    dot_size = 100
    plt.scatter(df['VAL'], df['ARO'], c=colors, edgecolors='k', s=dot_size)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(data_set[0])

    size = 10
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Neutral', markersize=size, markerfacecolor='yellow', markeredgecolor='k'),
                       Line2D([0], [0], marker='o', color='w', label='Happy', markersize=size, markerfacecolor='green', markeredgecolor='k'),
                       Line2D([0], [0], marker='o', color='w', label='Pain', markersize=size, markerfacecolor='red', markeredgecolor='k'),
                       Line2D([0], [0], marker='o', color='w', label='Strain', markersize=size, markerfacecolor='blue', markeredgecolor='k')]
    plt.legend(handles=legend_elements, loc='best')

    plt.grid(True)
    plt.show()


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
    aF4 = current_sample[56]
    bF4 = current_sample[57] #lub 58
    aF3 = current_sample[11]
    bF3 = current_sample[12] #lub 13
    aAF3 = current_sample[1]
    bAF3 = current_sample[2] #lub 3
    aAF4 = current_sample[66]
    bAF4 = current_sample[67] # lub 68
    tAF3 = current_sample[0]
    tAF4 = current_sample[65]


    gAF3 = current_sample[4]
    gAF4 = current_sample[69]
    gF3 = current_sample[14]
    gF4 = current_sample[59]

    #valence = aF4/bF4 - aF3/bF3
    #valence = np.mean([aF4-tAF4 , aF3-tAF3])
    #print(f"Valence: {valence}")
    
    #arousal = np.mean([bAF3/aAF3, bAF4/aAF4, bF3/aF3, bF4/aF4]) #>1 oznacza pobudzenie, stres, napięcie, wysoka koncentracja; <1 relaks, spokój, senność, medytacja
    #print(f"Arousal: {arousal}")

    valence = - (bF4 / bF3 * d(bF4, bF3) + gAF3 / gAF4 * d(gF4, gF3)) + aAF4 / aAF3 * d(aAF4, aAF3) + gF4 * 2
    arousal = gF4* 2

#    valence= 
#    arousal =

    
    arousal_scaled = arousal - 2.5

    state = calculate_state(valence, arousal)
    
    return valence, arousal, state



def print_state(valence, arousal_scaled, state, counter):
    if counter == N:
        print(f"VAL: {valence} ARO: {arousal_scaled}    STATE: {state}")
        counter = 0
    else:
        counter += 1
    
    return counter



def main():

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

    while True:
        # Returns a tuple (sample,timestamp) where sample is a list of channel values and timestamp is the capture time of the sample on the remote machine,
        # or (None,None) if no new sample was available
        sample, timestamp = inlet.pull_sample()
        if timestamp != None:
            #print(timestamp, sample)
            v, a, s = get_eeg_power_data(sample)
            printing_counter = print_state(v, a, s, printing_counter)


if __name__ == "__main__":
    main()
