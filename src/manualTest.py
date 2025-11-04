from NeuralNetwork import *
from settings import *
import json

def passThroughNN(nn: object, inputs: list):
    nn.forwardPass(inputs)
    return nn.layerOutput[nn.layers-1]

def main():
    with open("bias.json", "r") as f:
        bias = json.load(f)
    with open("weights.json", "r") as f:
        weights = json.load(f)
    
    nn = NN(bias, weights, LEARNING_RATE)

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

    idx = input(f"{inputs}\nPlease enter index of input: ")
    result = passThroughNN(nn, inputs[int(idx)])
    result = [round(x) for x in result]
    print(result)


if __name__ == '__main__':
    main()