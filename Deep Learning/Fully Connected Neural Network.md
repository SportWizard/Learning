# What is fully connected neural network?
Fully connected neural network (FCNN), also known as multi-layer perceptron (MLP) is a concept and a architecture used in [[Deep Learning]]. The concept of FCNN is every neurons in the previous layer is connected to every neuron in the next layer. The architecture of FCNN is a layer of neuron taking inputs, then passes it to the hidden layers, and lastly to the output layer (Usually [[Softmax]] for multi-class)

# What is neuron(s)?
In [[Deep Learning]], a neuron is an equation taking the form of a linear equation (each $w_i x_i$ is a new dimension)
$$
y = \vec{w}^\intercal \vec{x} + b = \sum_i w_i x_i + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$
that is applied with a [[Nonlinear activations]] function (e.g. ReLU)
$$
y = \phi(\vec{w}^\intercal \vec{x} + b) = \phi(\sum_i w_i x_i + b) = \phi(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)
$$
Then a layer of neurons will have the equation
$$
\vec{y} = \phi(W \vec{x} + \vec{b})
$$
$$
\vec{y}_i = \phi(\vec{w}_i^\intercal \vec{x} + b_i), \; \forall i \in \{1, 2, \cdots, m\}
$$
$y \in \mathbb{R}$ is the output of a neuron
$\vec{w} \in \mathbb{R}^n$; $w_i \in \mathbb{R}$ is the free parameters (weights)
$\vec{x} \in \mathbb{R}^n$; $x_i \in \mathbb{R}$ is the inputs (either from the data sample or the previous layer)
$b \in \mathbb{R}$ is the bias

$\phi(\cdot)$ is the [[Nonlinear activations]] function

$\vec{y} \in \mathbb{R}^m$ is a vector containing outputs from each neurons
$W = \begin{bmatrix}\vec{w}_1^\intercal \\ \vec{w}_2^\intercal \\ \vdots \\ \vec{w}_m^\intercal\end{bmatrix}$; $W \in \mathbb{R}^{m \times n}$ is a matrix containing $n$ free parameters (weights) of each neuron, where each free parameters (weights) has a dimension/size $m$
$W \vec{x} \in \mathbb{R}^{m}$ is a vector containing prediction
$\vec{b} \in \mathbb{R}^m$ is a vector containing biases

$\vec{y}_i \in \mathbb{R}$ is the output of a neuron
$\vec{w}_i \in \mathbb{R}^n$ is a free parameter (weight)
$\vec{w}_i \vec{x} \in \mathbb{R}$ is a prediction
$b_i \in \mathbb{R}$ is a bias

![[neuron.png]]
![[fcnn.png]]

# Objective/loss function
- Regression: [[Mean Square Error]] (other objective function can also be used)
- Classification: [[Cross Entropy Error]] (other objective function can also be used)

# Code
##### MLP classifier with Scikit-learn
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(activation="relu", batch_size=100, max_iter=20, solver="sgd", hidden_layer_sizes=(500,250), learning_rate_init=0.1, verbose=True)

mlp.fit(X, y)

mlp.predict(X)
```