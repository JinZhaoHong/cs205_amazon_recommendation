from keras.models import Model, Sequential
from keras.layers import Flatten, MaxPooling2D, Conv2D, Dense, Activation, Input
from keras import layers
from keras import models
from keras.utils import np_utils
from keras.datasets import mnist
import keras as keras
import keras.backend as K
import numpy as np
from os import listdir
from os.path import isfile, join
import sys



path = sys.argv[1]

X_train = []
Y_train = []

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

print("Loading data...")

for file in onlyfiles:
    file = join(path, file)
    f = open(file, "r", encoding='latin-1')
    
    line = f.readline()
    
    
    while line:
        try: 
            splitted = line.replace("(", "").replace(")", "") \
                  .replace("[", "").replace("]", "") \
                  .replace('\n', "").replace(" ", "").split(",")

            splitted = np.array(splitted).astype(float)
            X_train.append(splitted[:-1])
            Y_train.append(splitted[-1])
        except:
            pass

        line = f.readline() 

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train_ = np.zeros((X_train.shape[0], X_train[0].shape[0]))
Y_train_ = np.zeros((X_train.shape[0]))
for i in range(X_train.shape[0]):
    if len(X_train[i]) == len(X_train_[i]):
        X_train_[i] = X_train[i][:].tolist()
        Y_train_[i] = Y_train[i]
    
X_train = X_train_
Y_train = Y_train_

print("Building Dense NN...")


model = Sequential()
model.add(Dense(200, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1))

print(model.summary())


adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, batch_size=2048)

print("Predicting ratings...")

Y_predict = model.predict(X_train)
Y_predict.shape = Y_predict.shape[0]

MAE = np.sum(np.abs(Y_predict - Y_train)) / Y_predict.shape[0]
 
print("The Mean Absolute Error for Nerual Network Is: " + str(MAE))

i = int(float(X_train.shape[1])/2.0)
j = int(X_train.shape[1])

Y_predict2 = np.sum((X_train[:, :i] * X_train[:, i:j]), axis=1)
MAE2 = np.sum(np.abs(Y_predict2 - Y_train)) / Y_predict.shape[0]

print("The Mean Absolute Error for Simple Dot Product: " + str(MAE2))





