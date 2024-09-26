import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle, os, joblib

selfDirectory = os.getcwd()
picklesDirectory = os.path.abspath(os.path.dirname(__file__)) + "\\pickles"


def baseline_model(in_nr, out_nr):
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=in_nr, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(out_nr, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def mlp_fq(data, target_col, cols, export, happy_present, ts, display):
    if happy_present:
        states = ['HAPPY','NEUTRAL','PAIN','STRAIN']
    else:
        states = ['NEUTRAL','PAIN','STRAIN']
    dataset = data.values
    X = dataset[:, :len(cols)].astype(float)
    Y = dataset[:, len(cols)]

    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # Convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = to_categorical(encoded_Y)

    in_nr = X.shape[1]
    out_nr = dummy_y.shape[1]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=ts, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and fit the model
    clf = KerasClassifier(model=baseline_model, in_nr=in_nr, out_nr=out_nr, epochs=200, batch_size=5, verbose=0)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    factor = pd.factorize(data[target_col])
    definitions = factor[1]
    reversefactor = dict(zip(range(len(states)), definitions))

    # Convert predictions and true labels back from one-hot encoding
    y_test = [np.argmax(label) for label in y_test]
    y_pred = [np.argmax(label) for label in y_pred]
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    result = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(result)
    result1 = classification_report(y_test, y_pred)
    print('Classification Report:', )
    print(result1)
    result2 = accuracy_score(y_test, y_pred)
    print('Accuracy:', result2)
    if export:
        os.chdir(picklesDirectory)
        clf.model_.save("MLP.h5")
        joblib.dump(scaler, "MLP_scaler.pkl")
    ratio_label = str(int((1-ts)*100))+'/'+str(int(ts*100))

    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=result,display_labels=states)
        disp.plot()
        disp.ax_.set_title('MLP - test set '+ratio_label)
        plt.show()
