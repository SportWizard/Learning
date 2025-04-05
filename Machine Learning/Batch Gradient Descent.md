# What is batch gradient descent
Batch gradient descent (BGD) uses [[Gradient Descent]] to optimize the function, $arg \; min_{x} f(x)$ (arg means continuously)

# How it work?
- Uses all the training data to calculate the gradient of the objective/loss function with the current weights and calculates the average
- Update the weights using the equation
- adjust learning rate
- Repeat
# Equation
$$
W^* = W - lr \frac{\sum_{x \in \text{batch}} \nabla f_W(x)}{\text{size of batch}}, \text{batch} = \text{entire data samples}
$$ 
$W$ is the weights
$W^*$ is the updated weights
$lr$ is the learning rate (determine the step size)
$\nabla f(W)$ is the gradient

# Pros
- Final computed weight, $W^*$ will be accurate (preferred for small data samples)

# Cons
- Slow for large data samples

```python
import numpy as np
import jax.numpy as jnp
from jax import grad

def batch_gd(X_train, y_train, X_test, y_test, lr, annealing_rate, max_epochs):
    num_samples = X_train.shape[0]
    w = jnp.zeros(X_train.shape[1]) # Initalize weight

    # Initalization
    training_accuracy = np.zeros(max_epochs)
    test_accuracy = np.zeros(max_epochs)
    w_errors = np.zeros(max_epochs)

    # Run n epochs, where n = max_epochs
    for epoch in range(max_epochs):
        for index in range(num_samples):
            # Separate samples into mini-batches
            X = X_train[index]
            y = y_train[index]

            # Derive gradient and compute w*
            w_grad = grad(obj_func)(w, X, y)

            w = w - lr * w_grad

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