# What is softmax?
Softmax is a mathematical function that converts a vector of real numbers (logits) into a probability distribution

# Equation
$$
\vec{y} = softmax(\vec{x})
$$
$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}, \; \forall i \in \{1, 2, \cdots, n\}
$$
$\vec{y} \in [0, 1]^n$ (between 0 and 1 with dimension $n$) is the outputs
$\vec{x} \in \mathbb{R}^n$ is the inputs

$y_i \in \mathbb{R}$ is an output
$x_i \in \mathbb{R}$ is an input
$n$ is the number of neurons in the current layer

Note: $\sum_{i=1}^n y_i = 1$