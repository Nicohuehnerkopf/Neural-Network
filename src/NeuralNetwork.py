from math import tanh
from settings import *
import json

# sigmoid function ensures the non-linearity needed for a working Neural Network 
def sigmoid(x):
    return 0.5 * (1 + tanh(x / 2))

# the derivative is needed for the backpropagation algorithm
def derivativeSigmoid(x):
    return x * (1 - x)


class NN:
    def __init__(self, bias: list, weights: list, learningRate: float):
        self.bias = bias
        self.weights = weights
        self.learningRate = learningRate
        self.layerOutput = []   # stores the computed values of each neuron in each layer
        self.firstInputs = []   # stores the original inputs(needed for computing the weightschanges)
        self.unitErrors = []    # stores the computed errors from each neuron in each layer 
        self.layers = len(self.bias) # amount of layers
    

    def forwardPass(self, inputs: list):
        self.layerOutput = []   # emptying list to prevent pollution with several epochs 
        for l in range(self.layers):
            if l == 0:
                self.firstInputs = inputs.copy()
            else:
                inputs = self.layerOutput[l-1]  # this enables the reuse of code in the next section

            values = [] # list for ensuring right list architecture. Bad name, I can't think of something better
            for i in range(len(self.bias[l])):
                temp = []   # like values important for list architecture
                for j in range(len(inputs)):
                    temp.append(inputs[j] * self.weights[l][i][j])
                temp = sigmoid(sum(temp) + self.bias[l][i])
                values.append(temp)

            self.layerOutput.append(values.copy())


    def backprop(self, wantedOutputs: list):
        self.unitErrors = []    # emptying list to prevent pollution with several epochs
        for l in range(self.layers-1, -1, -1):
            delta = []  # list of the unit errors in the current layer
            if l == self.layers-1:
                for i in range(OUTPUTSIZE):
                    delta.append(derivativeSigmoid(self.layerOutput[l][i]) * (self.layerOutput[l][i] - wantedOutputs[i]))   # formula for computing the unit errors in the last layer
            
            else:
                for i in range(len(self.layerOutput[l])):   # similar to the forward pass we now propagate backwards through the network using the weighed sum of unit errors 
                    temp = []
                    for j in range(len(self.layerOutput[l+1])):
                        temp.append(self.unitErrors[0][j] * self.weights[l+1][j][i])
                    temp = sum(temp)
                    delta.append(derivativeSigmoid(self.layerOutput[l][i]) * temp)
            
            self.unitErrors.insert(0, delta.copy()) # inserting at 0 because we go backwards through the network


        weightchanges = []
        for l in range(self.layers):    # propagates forward through the Neural Network and computes the changes for the weights 
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
        

        for l in range(self.layers):    # Propagates forward through the Network adjusting every weight (could be implented in the block above for optimization)
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][i])):
                    self.weights[l][i][j] += weightchanges[l][i][j]
        
        

        biaschanges = []
        for l in range(self.layers):    # Propagates forward through the Network and computes the changes for the bias (could be implemented in the weightchanges block for optimization)
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