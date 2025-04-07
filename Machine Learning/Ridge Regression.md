# What is ridge regression?
Ridge regression is a linear model in [[Machine Learning]] that uses [[Supervised Learning]] to train. It is an extension of [[Linear Regression]] that incorporates regularization into its objective/loss function to prevent over-fitting

# What problem is ridge regression used for?
Ridge regression is primary used for regression problem. It can also do binary classification problem, but not great for it

# How does ridge regression work?
Ridge regression work similar to the equation $y = mx + b$, where $m$ in this is the weight vector, $\vec{w}$ and $b$ is the bias vector, $\vec{b}$.

**Training:**
	 During training, it uses gradient descent to adjust its weight vector such that it achieve the lowest loss
**Prediction:**
	Ridge regression uses $\vec{w}^\intercal \vec{x}$ to make its prediction

# Equation

$$
\hat{y} = \vec{w}^\intercal \vec{x}
$$
$\hat{y} \in \mathbb{R}$ is the output
$\vec{w} \in \mathbb{R}^n$ is a weight vector
$\vec{x} \in \mathbb{R}^n$ is an input vector

# Objective/loss function
$$
\vec{w}^* = \arg \underset{\vec{w}}{\min} \left[ \sum_{i=1}^N (\vec{w}^\intercal \vec{x}_i - y_i)^2 + \lambda \cdot \|\vec{w}\|_2^2 \right] = \arg \underset{\vec{w}}{\min} \left[ \|X\vec{w} - \vec{y}\|^2 + \lambda \cdot \|\vec{w}\|_2^2 \right]
$$
$\vec{w}^\intercal \vec{x}_i \in \mathbb{R}$ is the prediction
$y_i$ is the true value
$N$ is the total number of data points/samples

$X = \begin{bmatrix}\vec{x}_1^\intercal \\ \vec{x}_2^\intercal \\ \vdots \\ \vec{x}_m^\intercal\end{bmatrix}\in \mathbb{R}^{m \times n}$ is a matrix containing $m$ inputs vectors with dimension/size $n$
$X \vec{w} \in \mathbb{R}^m$ is a vector containing prediction
$\vec{y} \in \mathbb{R}^m$ is vector containing true values

$\lambda \in \mathbb{R}; \lambda \ge 0$ is how much does the regularization contributes to the objective/loss function (0 mean it is similar to linear regression's objective/loss function)
$\|\vec{w}\|_2^2$ is the L2 regularization
___
The objective/loss function is used to calculate the difference between the predicted value and the true value

# Regularization
The Regularization used in ridge regression is the L2 norm. Its purpose is to make some weights go near-zero, so some features doesn't contribute as much, resulting in a less complex model to prevent over-fitting

e.g.
$$
\vec{w}^\intercal \vec{x} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n
$$
$w_2$, $w_5$, $w_{24}$, ... could be near-zero

![[l2-regularization.png]]

# Closed-form solution
$$
\vec{w}^* = (X^\intercal X + \lambda \cdot \mathbf{I})^{-1} X^\intercal \vec{y}
$$
Note: $\mathbf{I}$ is the identity matrix

# Code
##### Mini-batch SGD for ridge regression
```python
import numpy as np
import jax.numpy as jnp
from jax import grad

class Optimizer:
    def __init__(self, lr, annealing_rate, batch_size, max_epochs, lam=0):
        self.lr = lr
        self.annealing_rate = annealing_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lam = lam

def obj_func(w, X, y, lam=0):
    # ||w^T x_i - y_i||^2 + lambda ||w||_2^2 = ||X w - y||^2 + lambda ||w||_2^2, where each vector in X are row vector, x_i^T, and N is the number of samples
    diff = X @ w - y
    lse = jnp.sum(diff * diff)
    regularization = lam * jnp.dot(w, w)

    return (lse + regularization) / X.shape[0] # Divide X.shape[0] to normalize

def ridge_regression_gd(X, y, op):
    num_samples = X.shape[0]
    w = jnp.zeros(X.shape[1]) # Initalize weight

    # Initalization
    lr = op.lr
    w_errors = np.zeros(op.max_epochs)

    # Run n epochs, where n = max_epochs
    for epoch in range(op.max_epochs):
        indices = np.random.permutation(num_samples) # Randomly shuffle data indices

        for batch_start in range(0, num_samples, op.batch_size):
            # Separate samples into mini-batches
            X_batch = X[indices[batch_start:batch_start + op.batch_size]]
            y_batch = y[indices[batch_start:batch_start + op.batch_size]]

            # Derive gradient and compute w*
            w_grad = grad(obj_func)(w, X_batch, y_batch, op.lam)

            w -= lr * w_grad / X_batch.shape[0] # mini-batch SGD uses the average gradient of the mini-batch

        # Learning curves
        w_errors[epoch] = obj_func(w, X, y) # Only calculate MSE by setting lambda to 0

        # Update learning rate (keep annealing_rate % of lr)
        lr *= op.annealing_rate

    return w,  w_errors

# Ridge regression
rigde_op = Optimizer(lr=0.02, annealing_rate=0.99, batch_size=50, max_epochs=20, lam=1) # 0 <= lambda <= posititve infinity
ridge_w, ridge_w_errors = ridge_regression_gd(X, y, rigde_op)

ridge_predict = X @ ridge_w

# Linear regression
# Since lam is not set to a value, meaning lam=0. Then this optimizer acts as a linear regression since the regularization control nothing to the objective function
linear_op = Optimizer(lr=0.02, annealing_rate=0.99, batch_size=50, max_epochs=20)
linear_w, linear_w_errors = ridge_regression_gd(X, y, linear_op)

linear_predict = X @ linear_w
```

##### Closed-form solution for ridge regression
```python
import numpy as np

lam = 1.0
w = np.linalg.inv(X.T @ X + lam * np.identity(X.shape[1])) @ X.T @ y

predict = X @ w
```