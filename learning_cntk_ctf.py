import numpy as np
import cntk as C
import time
from datetime import datetime

print(datetime.now())

from cntk.layers import Dense


#parameters

input_dim = 6000
out_classes = 3 
num_hidden_layers = 2
hidden_layers_dim = 10

learning_rate = 0.1
num_mb_iter = 100

#toDo: recondition the data so it's less distorted
#toDo: use less classes
#toDo: learn about learning rate parameter, it's way too stupid right now...
#toDo: plot all the data in matlab or something to see if it's even ok...
dataPath_train = 'C:/Users/ITRI/Documents/Programming/Csharp/tooldataNoupload/CTFdata/freqD3CS_train.ctf'
dataPath_test = 'C:/Users/ITRI/Documents/Programming/Csharp/tooldataNoupload/CTFdata/freqD3CS_test.ctf'

# Initialize the parameters for the trainer
minibatch_size_train = 72
num_samples_per_sweep = 72
num_sweeps_to_train_with = 100
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size_train

test_minibatch_size = 8
num_samples_test = 48
num_minibatches_to_test = int(num_samples_test / test_minibatch_size)

epochs = 10

#log every
num_train_prog = int(num_sweeps_to_train_with/10)



print("\n_______________________\n\nHello this is dog\n_______________________\n\n")


# input variables denoting the features and label data

feature = C.input_variable(input_dim) # C is CNTK
label = C.input_variable(out_classes)

# Instantiate the feedforward classification model
def create_model(features):
    with C.layers.default_options(init = C.layers.glorot_uniform(), activation = C.ops.relu):
            h = features
            for _ in range(num_hidden_layers):
                h = C.layers.Dense(hidden_layers_dim)(h)
            r = C.layers.Dense(out_classes, activation = None)(h)
            return r

z = create_model(feature)

# my_model = C.layers.Sequential ([
#                 Dense(hidden_layers_dim, activation=C.ops.relu),
#                 Dense(hidden_layers_dim, activation=C.ops.relu),
#                 Dense(out_classes, activation=None)])
# z = my_model(feature)

loss = C.losses.cross_entropy_with_softmax(z, label)
label_error = C.classification_error(z, label)

# Read a CTF formatted text using the CTF deserializer, from a .ctf file
def create_reader(path, is_training, input_dim, num_label_classes):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features   = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    )), randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


#logging training output
prog_printer = C.logging.ProgressPrinter(num_train_prog)

# Instantiate the trainer object to drive the model training
lr_schedule = C.learning_parameter_schedule(learning_rate, minibatch_size=num_mb_iter) #C.learners.IGNORE
learner = C.sgd(z.parameters, lr_schedule) # stochastic gradient descent/minibatch descent
trainer = C.Trainer(z, (loss, label_error), [learner], prog_printer)


# Create the reader to training data set
reader_train = create_reader(dataPath_train, True, input_dim, out_classes)

# Map the data streams to the input and labels.
input_map = {
    label  : reader_train.streams.labels,
    feature  : reader_train.streams.features
}

for k in range(0, int(epochs)):
    for i in range(0, int(num_minibatches_to_train)):

        # Read a mini batch from the training data file
        data = reader_train.next_minibatch(minibatch_size_train, input_map = input_map)

        trainer.train_minibatch(data)

    trainer.summarize_training_progress()


    #evaluation

    reader_test = create_reader(dataPath_test, False, input_dim, out_classes)

    test_input_map = {
        label  : reader_test.streams.labels,
        feature  : reader_test.streams.features,
    }

    # Test data for trained model
    
    test_result = 0.0

    for i in range(num_minibatches_to_test):

        # We are loading test data in batches specified by test_minibatch_size
        data = reader_test.next_minibatch(test_minibatch_size,
                                        input_map = test_input_map)

        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error
        print("Test minibatch {0}, average error: {1:.2f}%".format(i, eval_error*100))

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


print(datetime.now())

print('fin')