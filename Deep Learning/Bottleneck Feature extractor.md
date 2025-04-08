# What is bottleneck feature extractor
Bottleneck feature extractor (BFE) is an architecture and technique used in [[Deep Learning]] that implements non-linear dimension reduction on data samples using neural network
# How it works?
- The BFE is the encoder created using [[Fully Connected Neural Network]]), used to reduce the dimension of the data
- Once the data's dimension has been reduce, it will be used as the input of a model created using [[Fully Connected Neural Network]] for prediction
- The prediction will then be used in the objective function to calculate the loss and will optimize the BFE and model using [[Backpropagation]]
- Once training is finish, the model will be abandoned since we only need the BFE to reduce the dimension of the data

# Objective/loss function
Depends on the model used for prediction

![[bottleneck.png]]

Note: the bottleneck layer is the output from the dimension reduction

```python
# implement fully-connected neural networks using JAX and jax.grad()
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, random, device_put # device_put transfer data to a specific device (e.g. GPU)

class BFE():
    def __init__(self, optimizer="sgd", debug=0, struct1=[], struct2=[], activation="relu", loss="ce", lr=1.0, max_epochs=10, batch_size=10, random_state=1, init_range=1.0, annealing=1.0):
        self.optimizer = optimizer     # which optimizer is used to learn
        self.lr = lr                   # initial learning rate in SGD
        self.annealing = annealing     # annealing rate in SGD
        self.max_epochs = max_epochs   # max epochs in optimization
        self.batch_size = batch_size   # mini-batch size in SGD
        self.debug = debug             # whether print debugging info
        self.activation=activation     # activation function
        self.loss = loss               # the loss used for training objective
        self.random_state=random_state # random state
        self.init_range=init_range     # range for initializing weights

        self.struct1 = struct1         # Hidden layers for dimension reduction e.g. [100], [500, 200], [100,100,100]
        self.struct2 = struct2         # Hidden layers for classification

    # initialize internal struct/variables for input/output
    # X[N,d]: input features; Y[N,K]: 1-of-K one-hot vectors for output targets
    def initialization(self, X, Y):
        key = random.PRNGKey(self.random_state)

        input = X.shape[1]                # input dimension
        self.layers1 = len(self.struct1)  # number of hidden layers
        self.layers2 = len(self.struct2)

        # list for all weight matrices
        self.W_b1 = [0]*(self.layers1) # Bottleneck feature extractor
        self.W_b2 = [0]*(self.layers2+1) # Classification model (+1 is the output layer)

        # create weight matrices for all hidden layers

        # Bottleneck feature extractor
        for l in range(self.layers1):
            if l > 0 and self.struct1[l] >= self.struct1[l-1]:
                raise ValueError("struct1 must have decreasing layer sizes")

            output = self.struct1[l]
            self.W_b1[l] = device_put(4.90*(random.uniform(key,(input+1, output))-0.5)*self.init_range/jnp.sqrt(output+input)) # Generates a layer of neurons with random weights and biases from a uniform distribution
            input = output

        # Classification model
        for l in range(self.layers2):
            output = self.struct2[l]
            self.W_b2[l] = device_put(4.90*(random.uniform(key,(input+1, output))-0.5)*self.init_range/jnp.sqrt(output+input))
            input = output

        # create weight matrix for output layer
        output = Y.shape[1]
        self.W_b2[self.layers2] = device_put(4.90*(random.uniform(key,(input+1, output))-0.5)*self.init_range/jnp.sqrt(output+input))

    # forward pass to compute outputs for a mini-batch X
    # (refer to the box on page 166)
    # input  =>  X[B,d]: a batch of input vectors
    # return =>  y[B,K]
    def forward(self, W_b1, W_b2, X, return_features=False):

        # appending 1's to accomodate bias (see page 107)
        Z = jnp.hstack((X, jnp.ones((X.shape[0], 1), dtype=X.dtype)))

        # forward pass from all hidden layers

        # Bottleneck feature extractor
        for l in range(self.layers1):
            Z = jnn.relu(Z @ W_b1[l]) # Summation of w_i x_i + b
            Z = jnp.hstack((Z, jnp.ones((Z.shape[0], 1), dtype=Z.dtype)))

        # If it only require the bottleneck feature extractor
        if return_features:
            return Z

        # Classification model
        for l in range(self.layers2):
            Z = jnn.relu(Z @ W_b2[l]) # Summation of w_i x_i + b
            Z = jnp.hstack((Z, jnp.ones((Z.shape[0], 1), dtype=Z.dtype)))

        # forward pass for output layer
        l = self.layers2
        y = jnn.softmax(Z @ W_b2[l], axis=1)

        return y

    # compute the CE loss for a mini-batch
    # W_b[ ]: list for all weight matrices
    # X[B,d]: input features;
    # Y[B,K]: 1-of-K one-hot vectors for output targets
    def loss_ce_batch_bfe(self, W_b1, W_b2, X, Y):
        R = self.forward(W_b1, W_b2, X)

        return -jnp.mean(jnp.log(R[Y==1]))

    # use minibatch SGD to optimize (refer to Algorithm 8.8 on page 189)
    # X[N,d]: input features; Y[N,K]: 1-of-K one-hot vectors for output targets
    def sgd(self, X, Y):
        n = X.shape[0]      # number of samples

        lr = self.lr
        training_accuracy = np.zeros(self.max_epochs)
        #test_errors = np.zeros(self.max_epochs)
        w_errors = np.zeros(self.max_epochs)

        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n)  #randomly shuffle data indices
            for batch_start in range(0, n, self.batch_size):
                X_batch = X[indices[batch_start:batch_start + self.batch_size]]
                Y_batch = Y[indices[batch_start:batch_start + self.batch_size]]

                W_b1_grad, W_b2_grad = grad(self.loss_ce_batch_bfe, argnums=(0, 1))(self.W_b1, self.W_b2, X_batch, Y_batch) # argnums does the derivative of parameter 0, self.W_b1, with everything else as constant, then it does the derivative of parameter 1, self.W_b2, with everything else as constant

                # Apply backpropagation (backpropagation is used when there is multiple layers, such as FCNN, where each layer affect then next layer)
                for l in range(self.layers1):
                    self.W_b1[l] -= lr * W_b1_grad[l]

                for l in range(self.layers2+1):
                    self.W_b2[l] -= lr * W_b2_grad[l]

            # plot all learning curves
            w_errors[epoch] = self.loss_ce_batch_bfe(self.W_b1, self.W_b2, X, Y)

            Z = self.forward(self.W_b1, self.W_b2, X)
            train_label = np.argmax(Y, axis=1)
            train_res = np.argmax(Z, axis=1)
            training_accuracy[epoch] = np.count_nonzero(np.equal(train_res,train_label))/train_label.size

            if(self.debug):
                print(f"epoch = {epoch} (lr={lr:.2}): weight errors = {w_errors[epoch]:.5f}  training accuracy = {100*training_accuracy[epoch]:.2f}%")

            lr *= self.annealing

        return training_accuracy, w_errors

    # X[N,d]: input features; Y[N,K]: 1-of-K one-hot vectors for output targets
    def fit(self, X, Y):
        # initialize all weight matrices
        self.initialization(X, Y)

        X2 = device_put(X)
        Y2 = device_put(Y)

        training_accuracy, w_errors = self.sgd(X2, Y2)

        return training_accuracy, w_errors

    # X[N,d]: input features;
    # return: labels
    def predict(self, X):
        X2 = device_put(X)
        Y = self.forward(self.W_b1, self.W_b2, X2)
        return jnp.argmax(Y, axis=1)

    # X[N, d]: inputfeatures
    # return: feature reduction
    def extract_features(self, X):
        X2 = device_put(X)

        Z = self.forward(self.W_b1, [], X2, return_features=True)

        return Z[:, :-1] # Remove bias term
```

```python
# struct1: bottleneck feature extractor's hidden layers (each layer's neuron should be decreasing to reduce the dimension, where the last hidden layer is the bottleneck layer and it determines the new dimension/number of the feature)
# struct2: FCNN's hidden layers
bfe = BFE(struct1=[500, 250, 125, 64], struct2=[64, 32], debug=1, lr=0.1, annealing=0.98, batch_size=50, max_epochs=10)

training_accuracy, w_error = bfe.fit(bfe_X_train, bfe_y_train) # Train model
```

```python
# Feature reduction
reduced_X_train = bfe.extract_features(X_train)
reduced_X_test = bfe.extract_features(X_test)
```