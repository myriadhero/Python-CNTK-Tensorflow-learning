import numpy as np
import time


np.random.seed(int(time.time()))


#parameters

input_dim = 6000
out_classes = 2
num_hidden_layers = 1
hlayer_dim = 30
trainSetPercent = 0.6 #60% for training


data_scaling = 13000.0

num_train_iter = 500
epochs = 20


num_train_prog = 10
lrate = 0.01
num_mb_iter = 5

# import data
XYdata = np.genfromtxt('./ToolData/xData6001.csv', dtype='float', delimiter= ',')
dataShape = XYdata.shape


halfData = int(dataShape[0]/2)
trainSamples = int(halfData*trainSetPercent)
# i want to take a percantage of sharp and worn and have same amount

Xsharp = XYdata[:halfData,:input_dim]
Xworn = XYdata[halfData:,:input_dim]

tempYdata = XYdata[:,input_dim]
Ydata = np.zeros((dataShape[0],out_classes))

for x in range(dataShape[0]):
    Ydata[x, int(tempYdata[x])] = 1

Ysharp = Ydata[:halfData,:]
Yworn = Ydata[halfData:,:]

print(Ysharp)
print(Yworn)

XStrain = Xsharp[:trainSamples,:]
XStest = Xsharp[trainSamples:,:]
XWtrain = Xworn[:trainSamples,:]
XWtest = Xworn[trainSamples:,:]

YStrain = Ysharp[:trainSamples,:]
YStest = Ysharp[trainSamples:,:]
YWtrain = Yworn[:trainSamples,:]
YWtest = Yworn[trainSamples:,:]

print(XStrain.shape)
print(YWtrain)


# reassemble

#train data = [[YStrain , XStrain];
#              [YWtrain , XWtrain]]

#test data = [[YStest , XStest];
#             [YWtest , XWtest]]


# then shuffle
# then print into file