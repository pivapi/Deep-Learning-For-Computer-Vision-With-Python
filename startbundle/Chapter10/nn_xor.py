# python nn_xor.py

# import the necessary packages
import sys
sys.path.append("..")
from pyimagesearch.nn import NeuralNetwork
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our 2-2-1 nerual network and train it 
nn = NeuralNetwork([2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# Test the NN
print('[INFO]: Testing....')

for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result 
    # to our console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))
