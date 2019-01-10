# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt

import numpy as np
import sys
import os

import cntk as C



# Define the data dimensions
input_dim = 6000
num_output_classes = 2


print("\n_______________________\n\nHello this is dog\n_______________________\n\n")







# import data
XYdata = np.genfromtxt('./ToolData/xData6001.csv', dtype='float', delimiter= ',')
dataShape = XYdata.shape

# shuffle training data randomly
np.random.shuffle(XYdata)

# all data
Xdata = XYdata[:,:(dataShape[1]-1)]
tempYdata = XYdata[:,(dataShape[1]-1)]


# prep data for cntk
Ydata = np.zeros([dataShape[0],num_output_classes], dtype='float')

for x in range(dataShape[0]):
    Ydata[x, int(tempYdata[x])] = 1

# separate the training set
trainSetPercent = 0.6
trainSamples = int(dataShape[0]*trainSetPercent)

Xdata_train = Xdata[:trainSamples, :]
Ydata_train = Ydata[:trainSamples, :]
Xdata_eval = Xdata[trainSamples:, :]
Ydata_eval = Ydata[trainSamples:, :]

features = Xdata_train.astype(np.float32)
labels = Ydata_train.astype(np.float32)





# Helper function to generate a random data sample
#def generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
#    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
#    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)
#    X = X.astype(np.float32)
    # converting class 0 into the vector "1 0 0",
    # class 1 into vector "0 1 0", ...
#    class_ind = [Y==class_number for class_number in range(num_classes)]
#    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
#    return X, Y








# Create the input variables denoting the features and the label data. Note: the input
# does not need additional info on number of observations (Samples) since CNTK first create only
# the network tooplogy first
mysamplesize = 6
#features, labels = generate_random_data_sample(mysamplesize, input_dim, num_output_classes)







num_hidden_layers = 1
hidden_layers_dim = 30







# The input variable (representing 1 observation, in our example of age and size) x, which
# in this case has a dimension of 6000.
#
# The label variable has a dimensionality equal to the number of output classes in our case 2.

input = C.input_variable(input_dim)
label = C.input_variable(num_output_classes)







def create_model(features):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        last_layer = C.layers.Dense(num_output_classes, activation = None)

        return last_layer(h)

z = create_model(input)








loss = C.cross_entropy_with_softmax(z, label)






eval_error = C.classification_error(z, label)







# Instantiate the trainer object to drive the model training
learning_rate = 0.15
lr_schedule = C.learning_parameter_schedule(learning_rate)
learner = C.sgd(z.parameters, lr_schedule)
trainer = C.Trainer(z, (loss, eval_error), [learner])







# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]






# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error






# Initialize the parameters for the trainer
minibatch_size = 6
num_samples = 126
num_minibatches_to_train = num_samples / minibatch_size






# Run the trainer and perform model training
training_progress_output_freq = 10

#plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0,1000):
    trainer.train_minibatch({input : features, label : labels})
    print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

#for i in range(0, int(num_minibatches_to_train)):
    
#    inputdata = features[(i*minibatch_size):((i+1)*minibatch_size),:]
#    outputdata = labels[(i*minibatch_size):((i+1)*minibatch_size),:]
#    # Specify the input variables mapping in the model to actual minibatch data for training
#    trainer.train_minibatch({input : inputdata, label : outputdata})
#    batchsize, loss, error = print_training_progress(trainer, i,
#                                                     training_progress_output_freq, verbose=0)

#    if not (loss == "NA" or error =="NA"):
#        print("Iteration:\t" + str(i))
#        print("Loss:\t\t" + str(loss))
#        print("Error:\t\t" + str(error))
        #plotdata["batchsize"].append(batchsize)
        #plotdata["loss"].append(loss)
        #plotdata["error"].append(error)













