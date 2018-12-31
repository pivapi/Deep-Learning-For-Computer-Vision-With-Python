#import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

#sigmoid function
def sigmoid_activation(X):
    # compute the sigmoid activation value for a given input
    return 1 / (1 + np.exp(-X))

def predict(X, W):
    # take the dot product between out features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

# genrate a 2-class classification problem with 1.000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
# (1000, 3),第三列始终为1，bias
X = np.c_[X, np.ones((X.shape[0]))]


(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # take the dot product between our features 'X' and the weight
    # matrix 'W', then pass this value through our sigmoid activation
    # function, thereby giving us our predictions on the dataset
    # preds = sigmoid_activation(trainX.dot(W))
    preds = predict(trainX, W)

    # now the we have out predictions, we need to determine the
    # 'error', which isw the difference between our predictions and
    # the true values
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # the gradient descent update is the dot product between out
    # feature and the error of the predictions
    # (3, 1)
    gradient = trainX.T.dot(error)

    W += -args["alpha"] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# Evaluate the model
print('[INFO]: Evaluating....')
predictions = predict(testX, W)
print(classification_report(testY, predictions))

# Plot the classification (test) data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
# plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, s=30)

colorSign = []
for single in testY:
    if single == 1:
        colorSign.append('r')
    else:
        colorSign.append('b')

# plt.scatter(testX[:, 0], testX[:, 1], marker='o', s=30)
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=colorSign, s=30)

# Construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.title('Training Loss')
plt.plot(np.arange(0, args['epochs']), losses)
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()