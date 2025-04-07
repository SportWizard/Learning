# What is logistic regression?
Logistic regression is a linear model in [[Machine Learning]] that uses [[Supervised Learning]] to train. Its objective is to maximize the probability of correct classification in binary classification

# What problem is logistic regression used for?
Logistic regression is primary used for binary classification problem

# How does logistic regression work?
**Training:**
	 During training, it uses gradient descent to adjust its weight vector such that it achieve the highest correct classification
**Prediction:**
	Ridge regression uses $\text{sign}(\vec{w}^\intercal \vec{x})$ to make its prediction

# Equation
$$
\hat{y} = \text{sign}(\vec{w}^\intercal \vec{x})
$$
$\hat{y} \in \mathbb{R}$ is the output
$\vec{w} \in \mathbb{R}^n$ is a weight vector
$\vec{x} \in \mathbb{R}^n$ is an input vector

# Objective/loss function
$$
\begin{cases} 
\text{Correct classification}, & y_i \vec{w}^\intercal \vec{x}_i \gt 0 \\
\text{Misclassification}, & y_i \vec{w}^\intercal \vec{x}_i \lt 0
\end{cases}
$$
$$
s(x) = \frac{1}{1 + e^{-x}}
$$
$$
\vec{w}^* = \arg \underset{\vec{w}}{\max} \sum_{i=1}^N \ln s(y_i \vec{w}^\intercal \vec{x}_i) = \arg \underset{\vec{w}}{\max} \ln s(\vec{y} X \vec{w})
$$
Convert to minimize by adding a negative
$$
\vec{w}^* = \arg \underset{\vec{w}}{\min} - \sum_{i=1}^N \ln s(y_i \vec{w}^\intercal \vec{x}_i) = \arg \underset{\vec{w}}{\min} - \ln s(\vec{y} X \vec{w})
$$

$s(x)$ is the sigmoid function

$y_i$ is the true value
$\vec{w}$ is the weight vector
$\vec{x}$ is the input
$N$ is the total number of data points/samples

$X = \begin{bmatrix}\vec{x}_1^\intercal \\ \vec{x}_2^\intercal \\ \vdots \\ \vec{x}_m^\intercal\end{bmatrix}\in \mathbb{R}^{m \times n}$ is a matrix containing $m$ inputs vectors with dimension/size $n$
$X \vec{w} \in \mathbb{R}^m$ is a vector containing prediction
$\vec{y} \in \mathbb{R}^m$ is vector containing true values
___
The object function is used to calculate whether the predicted value and the true value have the same sign. If they have the same sign, the output would be positive and it is a correct classification. If they have opposite sign, the output would be negative and it is a misclassification.

![[sigmoid.png]]

# Differences between MCE and logistic regression
MCE learning focuses more on the boundary cases, while logistic regression yields faster convergence but is prone to outliers

# Code
##### Mini-batch SGD solution for logistic regression
```python
import numpy as np
import jax.numpy as jnp
from jax import grad

class Optimizer():
    def __init__(self, lr, annealing_rate, batch_size, max_epochs):
        self.lr = lr
        self.annealing_rate = annealing_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

# Objective function for logistic regression
def logistic_regression_func(w, X, y):
    # - 1/N Sigma^N_i=1 ln sigmoid(y_i w^T x_i)
    return - jnp.mean(jnp.log(sigmoid(y * (X @ w))))

def logistic_gd(X_train, y_train, X_test, y_test, op):
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
            w_grad = grad(logistic_regression_func)(w, X_batch, y_batch)

            w -= lr * w_grad

        # Learning curves
        w_errors[epoch] = logistic_regression_func(w, X, y) # Logisic regression loss function

		# Update learning rate (keep annealing_rate % of lr)
        lr *= op.annealing_rate

    return w, w_errors

# Logistic regression
logistic_op = Optimizer(lr=0.25, annealing_rate=0.99, batch_size=60, max_epochs=85)
logistic_w, logistic_w_errors = logistic_gd(X, y, logistic_op)

predict = jnp.sign(X @ w)
```