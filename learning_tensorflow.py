import tensorflow as tf
import numpy as np
import csv

#parameters



# import data
XYdata = np.genfromtxt('./ToolData/xData6001.csv', dtype='float', delimiter= ',')
dataShape = XYdata.shape

# shuffle training data randomly
np.random.shuffle(XYdata)

Xdata = XYdata[:,:(dataShape[1]-1)]
Ydata = XYdata[:,(dataShape[1]-1)]


print("TF version:\t" + (tf.VERSION))
print("Keras version:\t" + (tf.keras.__version__))

print(Ydata)

#model = tf.keras.Sequential()



#model.add(tf.keras.layers.Dense(30, activation='relu'))
#model.add(tf.keras.layers.Dense(2, activation= 'softmax'))




