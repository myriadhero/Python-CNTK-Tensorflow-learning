import numpy as np
import cntk as C
import time
from datetime import datetime

print(datetime.now())

from cntk.layers import Dense

#parameters

input_dim = 6000
out_classes = 2
num_hidden_layers = 1
hlayer_dim = 30
trainSetPercent = 0.6 #60% for training


data_scaling = 13000.0 # toDo: change normalization method

num_train_iter = 500
epochs = 20


num_train_prog = 10
lrate = 0.01
num_mb_iter = 5


np.random.seed(int(time.time()))


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
Ydata = np.zeros([dataShape[0],out_classes], dtype='float')

# map classes to 'vectorised' form that the library takes ie:
# if there are 3 classes, they need to be in the format of
# 0 = [1, 0, 0]; 1 = [0, 1, 0]; 2 = [0, 0, 1];...
for x in range(dataShape[0]):
    Ydata[x, int(tempYdata[x])] = 1

# separate the training set
# also perform data scaling for speeding up the convenrgence

trainSamples = int(dataShape[0]*trainSetPercent)

# toDo: get 60% from each class and then shuffle

Xdata_train = np.true_divide(Xdata[:trainSamples, :],data_scaling) 
Ydata_train = Ydata[:trainSamples, :]
Xdata_eval = np.true_divide(Xdata[trainSamples:, :],data_scaling)
Ydata_eval = Ydata[trainSamples:, :]


# the library likes float32 format instead of python's float64
features = Xdata_train.astype(np.float32)
labels = Ydata_train.astype(np.float32)
test_features = Xdata_eval.astype(np.float32)
test_labels = Ydata_eval.astype(np.float32)



# input variables denoting the features and label data

feature = C.input_variable(input_dim)
label = C.input_variable(out_classes)

# Instantiate the feedforward classification model
# toDo: implement a normalisation layer 

my_model = C.layers.Sequential ([
                Dense(hlayer_dim, activation=C.sigmoid),
                Dense(hlayer_dim, activation=C.relu),
                Dense(out_classes)])
z = my_model(feature)

ce = C.losses.cross_entropy_with_softmax(z, label)
pe = C.classification_error(z, label)

# logging module outputs training progress every num_train_prog
prog_printer = C.logging.ProgressPrinter(num_train_prog)

# learning rate is very important to get right, but I'm not sure how
# suggested way seems to be running multiple parameters and seeing which one does best
# minibatch_size changes learning rate every num_iterations
sgd_learner = C.sgd(z.parameters, lr = lrate, minibatch_size = num_mb_iter)
my_trainer = C.Trainer(z,(ce, pe),sgd_learner,prog_printer)

# iterations of gradient descent
for i in range(num_train_iter):
    my_trainer.train_minibatch({feature : features, label : labels})

my_trainer.summarize_training_progress()

# can manually change learning rate with code below
#lrate = 0.001
#lr_schedule = C.learning_parameter_schedule(lrate, minibatch_size=C.learners.IGNORE)
#my_trainer.parameter_learners[0].reset_learning_rate(lr_schedule)

# cross validation set or test set can be evaluated using trainer
avg_error = my_trainer.test_minibatch(
        {feature: test_features, label: test_labels})

my_trainer.summarize_test_progress()
print("\n error rate on an unseen minibatch %f \n" % avg_error)

#this is a trained model for the data set

model_output_file = "mModelZ1.dnn"
z.save(model_output_file)

print(datetime.now())