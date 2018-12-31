import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # Initialise the list of weight matrices, network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # Start looping from the index of the first layer but stop before we reach the last 2 layers
        for i in np.arange(0, len(layers) - 2):
            # Randomy initialise a weight matrix connecting the number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # The last 2 layers are a special case where the input connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # Return string that represents the network architecture
        return 'Neural Network: {}'.format('-'.join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # Compute the sigmoid activation value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # Compute the derivative of the sigmoid function assuming that 'x' has already been passed through the
        # sigmoid function
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        # Insert a column of 1's as the last entry of the feature matrix. This allows us the treat the bias as a
        # trainable parameter with the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop over the number of epochs
        for epoch in np.arange(0, epochs):
            # Loop over each data point and train the network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # Check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print('[INFO]: epoch={}, loss={:.5f}'.format(epoch+1, loss))

    def fit_partial(self, x, y):
        # Construct list of output activities for each layer as the data point flows through the network. The first
        # layer is just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEED-FORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # Feed forward the activation at the current layer by taking the dot product of the activation and the
            # weight matrix - called the 'net input' to the current layer
            net = A[layer].dot(self.W[layer])

            # The 'net output' is simply applying the sigmoid function to the net input
            out = self.sigmoid(net)

            # Add the net output to the list of activations
            A.append(out)

        # BACK-PROPAGATION:
        # Compute the difference between the 'prediction' (final net output in the activation list) and the true
        # target value
        error = A[-1] - y

        # Apply the chain rule to build a list of deltas. The first entry is simply the error of the output layer
        # times the derivative of the activation function for the ouput value
        D = [error * self.sigmoid_deriv(A[-1])]

        # Loop over the layers in reverse order (ignoring the last 2 layers)
        for layer in np.arange(len(A) - 2, 0, -1):
            # The delta for the current layer is equal to the delta of the 'previous layers' dotted with the weight
            # matrix of the current layer, followed by multiplying the delta by the derivative of the activation
            # function for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # Since we looped over the layer in reverse order we need to reverse the deltas
        D = D[::-1]

        # WEIGHT-UPDATE-PHASE:
        # Loop over the layers
        for layer in np.arange(0, len(self.W)):
            # Update the weights by taking the dot product of the layer activations with their respective deltas,
            # then multiplying this value by the learning rate and adding to the weight matrix
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        # Initialise the output prediction as the input features. This value will be (forward) propagated through the
        # network to obtain the final prediction
        p = np.atleast_2d(X)

        # Check to see if the bias column should be added
        if add_bias:
            # Insert a column of 1's as the last entry in the feature matrix
            p = np.c_[p, np.ones((p.shape[0]))]

        # Loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # Compute the output prediction
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # Return the predicted value
        return p

    def calculate_loss(self, X, targets):
        # Make predictions for the input data points then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # Return the loss
        return loss