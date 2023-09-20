# PROGRAMMING 1 CS445 #

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import time



# get the testing and training data for mnist
testdata = pd.read_csv("mnist_test.csv")
#testdata
traindata = pd.read_csv("mnist_train.csv")
#traindata
print(type(testdata),type(traindata))



# converted dataframe objects into numpy arrays
testdata=testdata.to_numpy()
traindata=traindata.to_numpy()
#print(type(traindata),type(testdata),traindata,testdata)
testdata=testdata.astype(float)
traindata =traindata.astype(float)



#print(traindata.shape)

#print(testdata.shape)
#print(type(traindata),type(testdata))
#print(traindata[0],"\n\n", testdata[0])

# downscale the data

for i in range(len(traindata)):
    for j in range(len(traindata[i])):
        if j != 0: # not the actual result
            traindata[i][j] = traindata[i][j]/255
            
for i in range(len(testdata)):
    for j in range(len(testdata[i])):
        if j != 0: # not the actual result
            testdata[i][j] = testdata[i][j]/255
        
for i in range(5):
    print(len(testdata[i]))
    #print(len(testdata[i][i])
print(len(testdata))


# we want a two layer neural network
# so we have: input layer, hidden layer, output layer
# n variable hidden units

# IDEA: make a train function
# after each training, update weights

# TODO: front phase first
    # 

# maybe 
# def train(n):   n is # hidden nodes
    # for x epochs:
        # for each row of data:
            # train model
            # update weights


# first element in x-vector and first in h-vector are the biases
# initialize w1 and w2 vectors to be random (-0.05,0.05)
# learning rate = 0.1
# momentum = 0.9


# w1 is of shape Input x (Hidden-1) since the first one is a bias
# w2 is of shape Hidden x Output
# hidden 
# bias unit = 1 and its weights will be first elemend in w1 and w2
# input vector is of shape 1x784

# target value tk for output value 
# k should be 0.9 if the input class is the kth class, and 0.1 if it isn't



alpha = 0.9 # momentum
eta = 0.1 # learning rate

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_weights(r,c):
    return np.random.uniform(-0.05,0.05, [r,c])
print(traindata.shape)
ex = traindata[:,0]
ex = ex



# input: #of_epochs, #hidden_nodes, momentum, learning_rate, data (either train or test)
def train(epoch,hidN,alpha,eta, data):

    
    w1 = initialize_weights(785,(hidN+1)) # 784 inputs , 20hidden
    w2 = initialize_weights((hidN+1), 10) # 20 hidden , 10 output possible

    w1momentum = 0.0
    w2momentum = 0.0
        
    ex = data[:,0]
    hidden = np.zeros(hidN+1)
    
    
    confmx = np.zeros([10,10])
    
    
    # store accuracy/epoch
    result = []
    

    for e in range(epoch): 
    
        print("EPOCH: ", e)
        
        for trainNum in range(len(data)):
            
            x = np.copy(data[trainNum])
            expected = ex[trainNum]

            
            x[0] = 1.0 # change the expected to be the bias
            
            
            hidden = np.matmul(x,w1)
            hidden = sigmoid(hidden)
            

            output = np.matmul(hidden, w2)
            output = sigmoid(output)

            #print(trainNum,expected,ex[trainNum],np.argmax(softmax(output)))

            confmx[int(expected)][np.argmax(output)] +=1

            target = np.empty(10)
            target.fill(0.1)
            target[int(expected)]=0.9

            #print(expected,target)
            #print(x)
            outE = np.zeros(len(output))
            outE = output*(1-output)*(target-output)
            #print(outE)



            #print(np.sum(outE))


            hidE = np.dot(outE, np.transpose(w2))
            temp = hidden*(1-hidden)
            hidE = temp*hidE


            t = np.outer(hidden, outE)
            w2delta = (eta*(t)+ (alpha*w2momentum))
            w2 = w2 + w2delta
            w2momentum = w2delta


            t = np.outer(x,hidE)
            w1delta = (eta*(t)+ (alpha*w1momentum))
            w1 = w1 +w1delta
            w1momentum = w1delta

            
        print(confmx)
        c = 0
        w = 0
        for i in range(10):
            for j in range(10):
                if i==j:
                    c+= confmx[i][j]
                else:
                    w+= confmx[i][j]
        print(c/(c+w),c,w)
        
        result.append(c/(c+w))
    return result


start = time.time()

x = train(20,50,0.5,0.1,traindata) 
#epochs, hidden, alpha, eta, data

end = time.time()

print("SECONDS: ", end-start)
print("RESULT: ",x)