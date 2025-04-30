import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from keras import Sequential
# from keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#import tensorflow as tf
print(tf.__version__)



#for creating linear layers
#from keras.layers import Dense


dataFrame = pd.read_csv('eyetracking_data.csv')

#separate features from classifications
dimensions = dataFrame[["mu_fixation","tau_fixation","mu_saccade_amp","tau_saccade_amp","mu_saccade_dur","tau_saccade_dur","mu_fixation_num","tau_fixation_num","mu_saccade_num","tau_saccade_num","mu_blink_num","tau_blink_num","mu_response_time","tau_response_time","mu_correct","tau_correct"]]
#defined using one hot encoding
classes = pd.get_dummies(dataFrame['CL_level'])
scaler = StandardScaler()
features = scaler.fit_transform(dimensions)


#create test and train sets
trainingFeatures, testFeatures, trainingClass, testClass = train_test_split(features, classes, test_size = .2, train_size=.8)


#model creation
#Sequential neural network
model = Sequential()

#layers of the network
model.add(Dense(8, activation='relu', input_shape=(trainingFeatures.shape[1],)))
model.add(Dense(units = 4, activation = 'softmax'))

model.add(Dense(3, activation='softmax'))


#metrics is just the information we want to know, in this case, how accurate the model is
#optimizer is gradient descent 
#loss is the error funciton: what you are trying to minimize
model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics= ['accuracy'])

def graphLosses(loss):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss) + 1), loss, marker='o', label='Training Loss')
    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def runTraining(model):
    #train the model
    #epochs = how many times it is trained// An epoch is an iteration over the entire x and y data provided
    #batch_size = how many points are passed in on each round (Number of samples per gradient update)
    return model.fit(trainingFeatures, trainingClass, epochs=25, batch_size=5, validation_split=0.2)


#graph the loss of each iteration
losses = runTraining(model).history['loss']



#how it performs on the test set
#returns the loss valuve and the metrics we defined above
print("now evaluate test")
score = model.evaluate(testFeatures, testClass)


print(score)

graphLosses(losses)
