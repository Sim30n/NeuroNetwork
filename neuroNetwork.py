# Petteri Särkkä 21.5.2020
"""
installing required dependencies:

pip install pandas
pip install sklearn
pip install numpy
pip install tensorflow
pip install matplotlib
pip install mlxtend

For Windows you might need to install and download visual studio 2015-2019 x86 or x64 from here:
https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads
"""
# For data formatting
import pandas as pd
import sklearn
from sklearn import model_selection
import numpy as np
# Neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# For plotting
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

# Formatting data for the training and testing set
def formatData(filename):
    df = pd.read_csv(filename, delimiter=";")
    yes_no = {"yes":1, "no":0}
    df.y = [yes_no[item] for item in df.y]
    # Dummies for categorical variables.
    df_y = df["y"]
    # Drop pdays and day and string y format
    df = df.drop(["pdays", "day", "y"], axis=1)
    # Dummies
    dummies = pd.get_dummies(data = df, columns = ["job","marital", "education",
                             "default", "housing", "loan", "contact", "month",
                             "poutcome"])
    df = pd.concat([dummies, df_y], axis=1)
    return df

# Formatting data for the custom test set. The original dataset also needed
# to get all the right columns.
def formatDataForTest(filename, filename2):
    df = pd.read_csv(filename, delimiter=";")
    if(filename2 != None):
        df2 = pd.read_csv(filename2, delimiter=";")
        df = pd.concat([df, df2], ignore_index=True)
    yes_no = {"yes":1, "no":0}
    df.y = [yes_no[item] for item in df.y]
    df_y = df["y"]
    # Drop pdays and day and string y format
    df = df.drop(["pdays", "day", "y"], axis=1)
    # Dummies
    dummies = pd.get_dummies(data = df, columns = ["job","marital", "education",
                             "default", "housing", "loan", "contact", "month",
                             "poutcome"])
    df = pd.concat([dummies, df_y], axis=1)
    len_df2 = len(df2)
    df = df.iloc[-len_df2:]
    return df

def neuroNetwork(df):
    X = df.iloc[:,0:49] #all the colums except "y"
    y = df.iloc[:,49] #"y" column
    # Training and test split 80/20.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y,
                                       test_size=0.2, random_state=0)
    # Building the model.
    model = Sequential()
    # Hidden layer with 49 inputs (starts from 0).
    model.add(Dense(40, input_dim=49, activation='sigmoid'))
    # Hidden layer, with sigmoid activation function.
    model.add(Dense(10,activation='sigmoid'))
    model.add(Dense(10,activation='sigmoid'))
    # Output layer, with sigmoid activation function.
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    x_train_keras = np.array(x_train)
    y_train_keras = np.array(y_train)
    y_train_keras = y_train_keras.reshape(y_train_keras.shape[0], 1)
    # Training the model.
    model.fit(np.array(x_train_keras),
              np.array(y_train_keras), epochs=250, verbose=0)
    scores = model.evaluate(np.array(x_test), np.array(y_test))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model, x_train, x_test, y_train, y_test

# Predict the result of chosen dataSet.
def predictDataSet(df, model):
    X = df.iloc[:,0:49] #all the colums except "y"
    y_pred = model.predict(X)
    pred = []
    for i in y_pred:
        if(i>0.5):
            pred.append(1)
        else:
            pred.append(0)
    return pred

def graphResults(model, x_test, y_test):
    # https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
    # http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.show()

dataSet = formatData("bank-full.csv")
# Original dataset and new custom test set.
testSet = formatDataForTest("bank-full.csv", "bank-test.csv")
model = neuroNetwork(dataSet)
predictedList = predictDataSet(testSet, model[0])
graphResults(model[0], model[2], model[4])
# Custom test set result.
list_index = 2
for i in predictedList:
    if(i==1):
        print("bank-test.csv line. %d predicted y =  yes" % list_index)
    else:
        print("bank-test.csv line. %d predicted y =  no" % list_index)
    list_index += 1
