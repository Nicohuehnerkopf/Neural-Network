import random as rn
from settings import *
import json

# this could be written using a list comprehension but it would make it more complicated
weights = []
for i in range(LAYERS):
    # rn.random() - 0.5 would also work
    if i == 0:
        weights.append(list(list(rn.uniform(-0.5, 0.5) for j in range(NUMINPUTS)) for j in range(HIDDEN_NEURONS[i])))   # randomizes the weights in the first layer based on input- and hiddenNeuron-size
    elif i == LAYERS-1:
        weights.append(list(list(rn.uniform(-0.5, 0.5) for j in range(HIDDEN_NEURONS[i-1])) for j in range(OUTPUTSIZE)))    # randomizes the weights in the last layer based on hiddenNeuron- and output-size
    else:
        weights.append(list(list(rn.uniform(-0.5, 0.5) for j in range(HIDDEN_NEURONS[i-1])) for j in range(HIDDEN_NEURONS[i]))) # randomizes the weights in the hidden layers based on the hiddenNeuron size

bias = []
for i in range(LAYERS):
    if i != LAYERS-1:
        bias.append(list(rn.uniform(-0.5, 0.5) for j in range(HIDDEN_NEURONS[i])))
    else:
        bias.append(list(rn.uniform(-0.5, 0.5) for j in range(OUTPUTSIZE)))


with open('weights.json', 'w') as f:
    json.dump(weights, f)

with open('bias.json', 'w') as f:
    json.dump(bias, f)