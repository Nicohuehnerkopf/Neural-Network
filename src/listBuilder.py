import random as rn
from settings import *
import json

weights = []
for i in range(LAYERS):
    if i == 0:
        weights.append(list(list(rn.uniform(-0.5, 0.5) for j in range(NUMINPUTS)) for j in range(HIDDEN_NEURONS[i])))       # rn.random() - 0.5 would also work
    elif i == LAYERS-1:
        weights.append(list(list(rn.uniform(-0.5, 0.5) for j in range(HIDDEN_NEURONS[i-1])) for j in range(OUTPUTSIZE)))
    else:
        weights.append(list(list(rn.uniform(-0.5, 0.5) for j in range(HIDDEN_NEURONS[i-1])) for j in range(HIDDEN_NEURONS[i])))

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