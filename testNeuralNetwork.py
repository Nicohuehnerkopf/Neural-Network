from NeuralNetwork import *
from settings import *
import random
import json


def passThroughNN(nn: object, inputs: list, outputs: list):
    nn.forwardPass(inputs)
    return nn.layerOutput[nn.layers-1]


def main():
    with open("bias.json", "r") as f:
        bias = json.load(f)
    with open("weights.json", "r") as f:
        weights = json.load(f)
    
    nn = NN(bias, weights, LEARNING_RATE)

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [0, 1, 1, 0]

    correct = 0
    incorrect = 0
    for epochs in range(10000):
        oneHotEncoding = []
        randidx = random.randint(0, len(inputs)-1)
        for i in range(OUTPUTSIZE):
            if i == outputs[randidx]:
                oneHotEncoding.append(1)
            else:
                oneHotEncoding.append(0)
        
        result = passThroughNN(nn, inputs[randidx], oneHotEncoding)
        result = [round(x) for x in result]

        isCorrect = 0
        for i in range(len(result)):
            if result[i] == oneHotEncoding[i]:
                isCorrect += 1
        if isCorrect == len(result):
            correct += 1
        else:
            incorrect += 1

        print(epochs)
    print(f"correct: {correct}")
    print(f"incorrect: {incorrect}")

if __name__ == '__main__':
    main()