import numpy as np
import time


np.random.seed(int(time.time()))


#parameters

input_dim = 6000 # can incorporate this into code, 
                    #but best to be sure and know from start
out_classes = 2 # same as above

trainSetPercent = 0.6 #60% for training

data_scaling = 13000.0 # have to also know this from data and model

finalFormatFile_train = './ToolData/YXFFData6001Train.txt'
finalFormatFile_test = './ToolData/YXFFData6001Test.txt'

# import data
XYdata = np.genfromtxt('./ToolData/xData6001.csv', dtype='float', delimiter= ',')
dataShape = XYdata.shape


halfData = int(dataShape[0]/2)
trainSamples = int(halfData*trainSetPercent)
# i want to take a percantage of sharp and worn and have same amount

# also perform data scaling on X
Xsharp = np.true_divide(XYdata[:halfData,:input_dim],data_scaling)
Xworn = np.true_divide(XYdata[halfData:,:input_dim], data_scaling)

tempYdata = XYdata[:,input_dim]
Ydata = np.zeros((dataShape[0],out_classes))

for x in range(dataShape[0]):
    Ydata[x, int(tempYdata[x])] = 1

Ysharp = Ydata[:halfData,:]
Yworn = Ydata[halfData:,:]

#print(Ysharp)
#print(Yworn)

XStrain = Xsharp[:trainSamples,:]
XStest = Xsharp[trainSamples:,:]
XWtrain = Xworn[:trainSamples,:]
XWtest = Xworn[trainSamples:,:]

YStrain = Ysharp[:trainSamples,:]
YStest = Ysharp[trainSamples:,:]
YWtrain = Yworn[:trainSamples,:]
YWtest = Yworn[trainSamples:,:]

#print(XStrain.shape)
#print(YWtrain)


# reassemble
#train data = [[YStrain , XStrain];
#              [YWtrain , XWtrain]]
YXStrain = np.concatenate((YStrain, XStrain), axis =1)
YXWtrain = np.concatenate((YWtrain, XWtrain), axis =1)

YXtrain = np.concatenate((YXStrain,YXWtrain))

#test data = [[YStest , XStest];
#             [YWtest , XWtest]]
YXStest = np.concatenate((YStest, XStest), axis =1)
YXWtest = np.concatenate((YWtest, XWtest), axis =1)

YXtest = np.concatenate((YXStest,YXWtest))

# then shuffle
np.random.shuffle(YXtrain)
np.random.shuffle(YXtest)

# then print into file
# f.write('|labels {} |features {}\n'.format(label_str, feature_str))
def saveFormattedTxt(outFilePath, npArray, outClasses):
    #print(npArray)
    num_samp = npArray.shape[0]

    with open(outFilePath, 'w') as f:
        for row in range(num_samp):
            YString = " ".join(map(str, npArray[row,:outClasses].astype(int)))
            XString = " ".join(map(str, npArray[row,outClasses:]))

            f.write('|labels {} |features {}\n'.format(YString, XString))
            #print(row)
        print('finished writing file ' + outFilePath)

saveFormattedTxt(finalFormatFile_test, YXtest, out_classes)
saveFormattedTxt(finalFormatFile_train, YXtrain, out_classes)

print('fin')