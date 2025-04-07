# What is linear regression?
Linear regression is a linear model in [[Machine Learning]] that uses [[Supervised Learning]] to train

# What problem is linear regression used for?
Linear regression is primary used for regression problem. It can also do binary classification problem, but not great for it

# How does linear regression work?
Linear regression work similar to the equation $y = mx + b$, where $m$ in this is the weight vector, $\vec{w}$ and $b$ is the bias.

**Training:**
	 During training, it uses gradient descent to adjust its weight vector such that it achieve the lowest loss
**Prediction:**
	Linear regression uses $\vec{w}^\intercal \vec{x}$ to make its prediction

# Equation
$$
\hat{y} = \vec{w}^\intercal \vec{x}
$$
$\hat{y} \in \mathbb{R}$ is the output
$\vec{w} \in \mathbb{R}^n$ is a weight vector
$\vec{x} \in \mathbb{R}^n$ is an input vector

![[linear-regression.png]]

# Objective/loss function
$$
\vec{w}^* = \arg \underset{\vec{w}}{\min} \sum_{i=1}^N (\vec{w}^\intercal \vec{x}_i - y_i)^2 = \arg \underset{\vec{w}}{\min} \|X\vec{w} - \vec{y}\|^2
$$
$\vec{w}^\intercal \vec{x}_i \in \mathbb{R}$ is the prediction
$y_i$ is the true value
$N$ is the total number of data points/samples

$X = \begin{bmatrix}\vec{x}_1^\intercal \\ \vec{x}_2^\intercal \\ \vdots \\ \vec{x}_m^\intercal\end{bmatrix}\in \mathbb{R}^{m \times n}$ is a matrix containing $m$ inputs vectors with dimension/size $n$
$X \vec{w} \in \mathbb{R}^m$ is a vector containing prediction
$\vec{y} \in \mathbb{R}^m$ is vector containing true values
___
The objective/loss function is used to calculate the difference between the predicted value and the true value

# Close-form solution
Close-form solution is used to calculate the exact weight, $w^*$, that will achieve the best result. It is preferred is the required computation is small. Otherwise, gradient descent is preferred even if it might not be as accurate as close-form solution

$$
\vec{w}^* = (X^\intercal X)^{-1} X^\intercal \vec{y} \text{ with } X^\intercal X \in \mathbb{R}^{n \times n}
$$
Note: sometimes SGD might be preferred to derive $w^*$ iteratively to avoid matrix inversion

# Binary classification
- Convert one class to -1 and the other class to +1
- Train the linear regression model
- Once training is finish, use the sign of the predict to classify the input as -1 or +1 (use numpy.sign() to convert to -1 and +1)
- ![[linear-regression-classification.png]]

# Code
##### Linear regression with mini-batch SGD:
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

def obj_func(w, X, y):
	diff = X @ w - y # prediction difference
	return jnp.sum(diff*diff)/X.shape[0]

# X[N,d]: input features; y[N]: output targets; op: hyper-parameters for optimzer
def linear_regression_gd(X, y, op):
	n = X.shape[0] # number of samples
	w = jnp.zeros(X.shape[1]) # initialization

	lr = op.lr
	errors = np.zeros(op.max_epochs)
	for epoch in range(op.max_epochs):
		indices = np.random.permutation(n) # randomly shuffle data indices
		for batch_start in range(0, n, op.batch_size):
			X_batch = X[indices[batch_start:batch_start + op.batch_size]]
			y_batch = y[indices[batch_start:batch_start + op.batch_size]]

			# compute gradients via auto-grad for a whole mini-batch
			w_grad = grad(obj_func)(w, X_batch, y_batch)
		
			w -= lr * w_grad / X_batch.shape[0]

		# Learning curves
		errors[epoch] = obj_func(w, X, y)

		# Update learning rate (keep annealing_rate % of lr)
		lr *= op.annealing_rate

	return w, np.array(errors)

# Linear regression
op = Optimizer(lr=0.005, annealing_rate=0.99, batch_size=30, max_epochs=500)
w, errors = linear_regression_gd(X, y, op)

predict = X @ w # Each row in X is a vector and w is a 1D vector (could be row or column vector)
```

##### Linear regression using its closed-form solution:
```python
import numpy as np

w = np.linalg.inv(X.T @ X) @ X.T @ y
predict = X @ w # Each row in X is a vector and w is a 1D vector (could be row or column vector)
```