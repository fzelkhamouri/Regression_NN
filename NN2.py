import numpy as np

class Neural_Network(object):
    def __init__(self):
        # Parametters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        # Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) weight matrix size
        print(self.W1)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix size
        print(self.W2)
        
    def forward(self,X):
        # Forward propagation through the network
        self.z = np.dot(X, self.W1) # dot products of input with firt set of 2x3 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer z2 and second set of 3x1 weights
        o = self.sigmoid(self.z3) # Final activation Function
        return o
    
    def sigmoid(self, s):
        # activation function
        return 1/(1 + np.exp(-s))
    
    def sigmoidPrime(self, s):
        # Derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
        # Backward propagation through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
        
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt = "%s")
        np.savetxt("w2.txt", self.W2, fmt = "%s")
        
    def predict(self):
        print("Predictive Data Based on trained Weights: ")
        print("Input (Scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))
        
        
        
# X = (hours studying, hours sleeping), y = score in test, xPredicted = 4 hours studying 

X = np.array(([2,9], [1,5], [3,6]), dtype = float)
y = np.array(([92],[86],[89]), dtype = float)
xPredicted = np.array(([4,8]), dtype = float)

# scale units
print(X)

X = X/np.amax(X, axis = 0) # Maximum of X array
print(X)

print(xPredicted)
xPredicted = xPredicted/np.amax(xPredicted, axis = 0)
print(xPredicted)

y = y/100 # max test score


NN = Neural_Network()

for i in range(10): # Trains the NN 1000 times
    print("#" + str(i) + "\n")
    print("Input (scaled): \n" + str(X))
    print("Actual output: \n" + str(y))
    print("Predicted output: \n" + str(NN.forward(X)))
    print("Loss: \n" + str(np.mean(np.square(y-NN.forward(X)))))
    print("\n")
    NN.train(X, y)



NN.saveWeights()
NN.predict()




        
        
    