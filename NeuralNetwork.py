from random import shuffle
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.size = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                       for x, y in zip(layers[:-1], layers[1:])]

    def propagate(self, a):
        """Input an array of size 784 and recieve answer for which number the neural network finds.""" 
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backPropagate(self, x, y):
        """Takes the returned vector and the desired vector and returns chanhes to be made to weights and biases."""
        
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.costDerivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.size):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (nablaB, nablaW)

    def stochasticGradientDescent(self, trainingData, epochs, batchSize, learningRate, testingData):
        """Trains the network by slow incremental changing of weights and biases. """
        nTest = len(testingData)
        n = len(trainingData)

        for j in range(epochs):
            shuffle(trainingData)
            batches = [trainingData[k:k + batchSize] for k in range(0, n, batchSize)]

            for batch in batches:
                self.updateBatch(batch, learningRate)
            
            print("Epoch {0}: {1} / {2}     {3}%".format(j, self.evaluate(testingData), nTest, self.evaluate(testingData)/nTest))

    def updateBatch(self, batch, learningRate):
        """Does the changing of weights and bisases by computing difference between desired and recieved."""
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            deltaNablaB, deltaNablaW = self.backPropagate(x, y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]

        self.weights = [w - (learningRate / len(batch)) * nw
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (learningRate / len(batch)) * nb
                        for b, nb in zip(self.biases, nablaB)]

    def evaluate(self, testingData):
        """Checks the 10000 testcases agains the network and returns the amount of successes."""
        x, y = testingData[0]
        testingResults = [(np.argmax(self.propagate(x)), y)
                            for x, y in testingData]
        return sum(int(x == y) for (x, y) in testingResults)

    def costDerivative(self, outputActivations, y):
        return (outputActivations - y)

def sigmoid(z):
    """Used for determining what the value of a neuron should be."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Helper function for backpropagation"""
    return sigmoid(z) * (1 - sigmoid(z))