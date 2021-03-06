import numpy as np
import cntk as C
import time

from cntk.layers import Dense



np.random.seed(0)


print("\n_______________________\n\nHello this is dog\n_______________________\n\n")

num_output_classes = 2
input_dim = 6000


# import data
XYdata = np.genfromtxt('./ToolData/xData6001.csv', dtype='float', delimiter= ',')
dataShape = XYdata.shape

# shuffle training data randomly
np.random.shuffle(XYdata)

# all data
Xdata = np.true_divide(XYdata[:,:(dataShape[1]-1)], 13000.0)
Ydata = XYdata[:,(dataShape[1]-1)]

features = Xdata.astype(np.float32)
labels = Ydata.astype(np.float32)

# load model
model_file = "mModelZ1.dnn"
z = C.load_model(model_file)
out = C.softmax(z) # assigns probability based values

# evaluate model, output predictions
predictions = out.eval({out.arguments[0]: features})

# assign class to each prediction
predict_label = np.zeros((predictions.shape[0],1))
sum = 0
for row in range(predictions.shape[0]):
    predict_label[row] = np.argmax(predictions[row,:])
    
    sum = sum + np.abs(predict_label[row] - Ydata[row])
    
print(100*(1 - sum/predictions.shape[0]))
    

