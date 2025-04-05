# What is mini-batch stochastic gradient descent
Mini-batch stochastic gradient descent (min-batch SGD) uses [[Gradient Descent]] to optimize the function, $arg \; min_{x} f(x)$ (arg means continuously). It does the same thing as SGD but with mini-batch instead of a single data sample

# How it work?
- Randomly shuffle all training data (only need to shuffle the index during coding)
- Put training data into batches
- Select the a training batch and calculate the gradient of the objective/loss function with the current weights for each training data in the batch and calculates the average
- Update the weights using the equation
- adjust learning rate
- Repeat
# Equation
$$
W^* = W - lr \frac{\sum_{x \in \text{mini-batch}} \nabla f_W(x)}{\text{size of mini-batch}}
$$
$W$ is the weights
$W^*$ is the updated weights
$lr$ is the learning rate (determine the step size)
$\nabla f(W)$ is the gradient

# Pros
- Fast training when there is a fair large amount of data samples

# Cons
- Final computed weight, $W^*$ will be close but not exact

```python
import numpy as np
import jax.numpy as jnp
from jax import grad

def mini_batch_sgd(X_train, y_train, X_test, y_test, lr, annealing_rate, max_epochs, batch_size):
    num_samples = X_train.shape[0]
    w = jnp.zeros(X_train.shape[1]) # Initalize weight

    # Initalization
    training_accuracy = np.zeros(max_epochs)
    test_accuracy = np.zeros(max_epochs)
    w_errors = np.zeros(max_epochs)

    # Run n epochs, where n = max_epochs
    for epoch in range(max_epochs):
        indices = np.random.permutation(num_samples) # Randomly shuffle data indices

        for batch_start in range(0, num_samples, batch_size):
            # Separate samples into mini-batches
            X_batch = X_train[indices[batch_start:batch_start + batch_size]]
            y_batch = y_train[indices[batch_start:batch_start + batch_size]]

            # Derive gradient and compute w*
            w_grad = grad(obj_func)(w, X_batch, y_batch)

            w = w - lr * w_grad / X_batch.shape[0] # mini-batch SGD uses the average gradient of the mini-batch

        # Learning curves
        w_errors[epoch] = obj_func(w, X_train, y_train)

        train_pred = np.sign(X_train @ w)
        training_accuracy[epoch] = np.count_nonzero(np.equal(train_pred, y_train)) / y_train.size

        test_pred = np.sign(X_test @ w)
        test_accuracy[epoch] = np.count_nonzero(np.equal(test_pred, y_test)) / y_test.size

        # Update learning rate (keep annealing_rate % of lr)
        lr *= annealing_rate
    
    return w, training_accuracy, test_accuracy, w_errors
```

Note: an epoch is when it uses all the training data once