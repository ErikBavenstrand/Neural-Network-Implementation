import pickle
import sys

from mnist import MNIST
from NeuralNetwork import *
import numpy as np
from PIL import Image

def vectorizeResult(x):
    e = np.zeros((10, 1))
    e[x] = 1.0
    return e

def getImageArray(fileName):
    ls = []
    for p in np.invert(Image.open(fileName).convert('L')).ravel():
        ls.append([p])
    return np.array(ls)/255

def createNeuralNetwork(layers, name):
    layers = list(map(int, layers))
    NN = NeuralNetwork(layers)
    data = MNIST('Data')
    trainingInput, trainingOutput = data.load_training()
    testingInput, testingOutput = data.load_testing()

    trainingInput = np.array(trainingInput)/255
    testingInput = np.array(testingInput)/255

    trainingInput = [np.reshape(x, (layers[0], 1)) for x in trainingInput]
    trainingOutput = [vectorizeResult(x) for x in trainingOutput]
    trainingData = list(zip(trainingInput, trainingOutput))

    testingInput = [np.reshape(x, (layers[0], 1)) for x in testingInput]
    testingData = list(zip(testingInput, testingOutput))

    NN.stochasticGradientDescent(trainingData, 50, 30, 2.0, testingData)

    binaryFile = open(name, mode='wb')
    neuralNetwork = pickle.dump(NN, binaryFile)
    binaryFile.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Creating a neural network...")
        createNeuralNetwork(sys.argv[1:-1], sys.argv[-1])
        print("Done")
    else:
        fileName = sys.argv[1]
        NN = pickle.load(open(fileName, 'rb'))

        while True:
            numberFile = input("What file would you like to read? ")
            if numberFile == '':
                break
            elif numberFile == 'all':
                for i in range(10):
                    f = str(i) + '.png'
                    val = np.argmax(NN.propagate(getImageArray(f)))
                    print("written number: {0}. Network finds a: {1}. {2}".format(i, val, val == i))

            else:
                numberFile += '.png'
                print(np.argmax(NN.propagate(getImageArray(numberFile))))
