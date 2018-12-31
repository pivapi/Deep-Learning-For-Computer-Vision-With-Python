# python perceptron_and.py

import sys
sys.path.append("..")
from pyimagesearch.nn import Perceptron
import numpy as np

# construct the AND dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)


print("[INFO] testing perceptron...")

for (x, target) in zip(X, y):

    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
