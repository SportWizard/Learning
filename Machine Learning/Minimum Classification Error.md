# What is minimum classification error?
Minimum classification error (MCE) is an objective/loss function used in [[Machine Learning]] to minimize the probability of misclassification in binary classification

# What problem requires MCE?
MCE is primary used in binary classification

# How does MCE work?
**Training:**
	 During training, it uses gradient descent to adjust its weight vector such that it achieve the lowest misclassification
**Prediction:**
	Ridge regression uses $\text{sign}(\vec{w}^\intercal \vec{x})$ to make its prediction

# Equation (for classification)
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
\vec{w}^* = \arg \underset{\vec{w}}{\min} \sum_{i=1}^N s(-y_i \vec{w}^\intercal \vec{x}_i) = \arg \underset{\vec{w}}{\min} s(-\vec{y} X \vec{w})
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

The negative in front of $y_i \vec{w}^\intercal \vec{x}_i$ can be thought of to accommodate $e^{-x}$ 's negative sign in the sigmoid function. If $y_i \vec{w}^\intercal \vec{x}_i$ is a large positive number, and by multiplying the negative sign, it will be a large negative number. Then, $\frac{1}{1 + e^{-x}} \approx 0$, which is great since we are trying to minimize the loss. If $y_i \vec{w}^\intercal \vec{x}_i$ is a large negative number, and by multiplying the negative sign, it will be a large positive number. Then, $\frac{1}{1 + e^{-x}} \approx 1$, which is not great since we are trying to minimize it the loss

A simplified way to think about it is $y_i \vec{w}^\intercal \vec{x}_i$ is the correct classification and $-y_i \vec{w}^\intercal \vec{x}_i$ is misclassification

![[sigmoid.png]]

# Differences between MCE and logistic regression
MCE learning focuses more on the boundary cases, while logistic regression yields faster convergence but is prone to outliers

# Code
##### Mini-batch SGD solution for MCE
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

# Objective function for MCE
def mce_func(w, X, y):
    # Sigma^N_i=1 sigmoid(-y_i w^T x_i)
    return jnp.mean(sigmoid(-y * (X @ w))) # mean = sum + divide by sample size, where the dividing by sample size is used to normalize

def mce_gd(X, y, op):
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
            w_grad = grad(mce_func)(w, X_batch, y_batch)

            w = w - lr * w_grad

        # Learning curves
        w_errors[epoch] = mce_func(w, X, y) # MCE loss function

		# Update learning rate (keep annealing_rate % of lr)
        lr *= op.annealing_rate

    return w, w_errors

# MCE
mce_op = Optimizer(lr=0.25, annealing_rate=0.99, batch_size=60, max_epochs=85)
mce_w, mce_w_errors = mce_gd(X, y, mce_op)

predict = jnp.sign(X @ w)
```