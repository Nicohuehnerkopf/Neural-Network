from NeuralNetwork import *
from settings import *
import random
import json



def passThroughNN(nn: object, inputs: list, outputs: list):
    nn.forwardPass(inputs)
    nn.backprop(outputs)

def main():
    with open("bias.json", "r") as f:
        bias = json.load(f)
    with open("weights.json", "r") as f:
        weights = json.load(f)

    nn = NN(bias, weights, LEARNING_RATE)
    
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [0, 1, 1, 0]

    for epochs in range(100000):
        oneHotEncoding = []
        randidx = random.randint(0, len(inputs)-1)
        oneHotEncoding.append(outputs[randidx])
        
        passThroughNN(nn, inputs[randidx], oneHotEncoding)
        print(epochs)
    
    with open("bias.json", "w") as f:
        json.dump(nn.bias, f)
    with open("weights.json", "w") as f:
        json.dump(nn.weights, f)

if __name__ == '__main__':
    main()