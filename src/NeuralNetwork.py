from math import e, tanh, log
from settings import *
import json

def sigmoid(x):
    return 0.5 * (1 + tanh(x / 2))

def derivativeSigmoid(x):
    return x * (1 - x)

class NN:
    def __init__(self, bias: list, weights: list, learningRate: float):
        self.bias = bias
        self.weights = weights
        self.learningRate = learningRate
        self.layerOutput = []
        self.firstInputs = []
        self.unitErrors = []
        self.layers = len(self.bias)
        self.Loss = -1
    

    def forwardPass(self, inputs: list):
        self.layerOutput = []
        for l in range(self.layers):
            if l == 0:
                self.firstInputs = inputs.copy()
            else:
                inputs = self.layerOutput[l-1]

            values = []
            for i in range(len(self.bias[l])):
                temp = []
                for j in range(len(inputs)):
                    temp.append(inputs[j] * self.weights[l][i][j])
                temp = sigmoid(sum(temp) + self.bias[l][i])
                values.append(temp)

            self.layerOutput.append(values.copy())


    def backprop(self, wantedOutputs: list):
        self.unitErrors = []
        for l in range(self.layers-1, -1, -1):
            delta = []
            if l == self.layers-1:
                for i in range(OUTPUTSIZE):
                    delta.append(derivativeSigmoid(self.layerOutput[l][i]) * (self.layerOutput[l][i] - wantedOutputs[i]))
            
            else:
                for i in range(len(self.layerOutput[l])):
                    temp = []
                    for j in range(len(self.layerOutput[l+1])):
                        temp.append(self.unitErrors[0][j] * self.weights[l+1][j][i])
                    temp = sum(temp)
                    delta.append(derivativeSigmoid(self.layerOutput[l][i]) * temp)
            
            self.unitErrors.insert(0, delta.copy())


        weightchanges = []
        for l in range(self.layers):
            if l == 0:
                values = []
                for i in range(len(self.weights[l])):
                    temp = []
                    for j in range(len(self.weights[l][i])):
                        temp.append(-self.learningRate * self.unitErrors[l][i] * self.firstInputs[j])
                    values.append(temp.copy())
            
            else:
                values = []
                for i in range(len(self.weights[l])):
                    temp = []
                    for j in range(len(self.weights[l][i])):
                        temp.append(-self.learningRate * self.unitErrors[l][i] * self.layerOutput[l-1][j])
                    values.append(temp.copy())
            weightchanges.append(values.copy())
        

        for l in range(self.layers):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][i])):
                    self.weights[l][i][j] += weightchanges[l][i][j]
        
        

        biaschanges = []
        for l in range(self.layers):
            temp = []
            for i in range(len(self.bias[l])):
                temp.append(-self.learningRate * self.unitErrors[l][i])
            biaschanges.append(temp.copy())

        for l in range(self.layers):
            for i in range(len(self.bias[l])):
                self.bias[l][i] += biaschanges[l][i]
        

    
def main():
    with open("bias.json", "r") as f:
        bias = json.load(f)
    with open("weights.json", "r") as f:
        weights = json.load(f)
    
    nn = NN(bias, weights, LEARNING_RATE)
    nn.forwardPass([1, 1])
    nn.backprop([0, 0])

if __name__ == '__main__':
    main()