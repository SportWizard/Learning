# What is nonlinear activation?
Nonlinear activation are mathematical functions used in [[Deep Learning]]. It is applied to the outputs of neurons in a neural network to introduce nonlinearity, enabling the network to model complex relationships in data

# Examples
Sigmoid ($\text{Domain} \in \mathbb{R} \text{; Range} \in (0, 1)$ - monotonic increasing, differentiable everywhere):
$$
s(x) = \frac{1}{1 + e^{-x}}
$$
tanh ($\text{Domain} \in \mathbb{R} \text{; Range} \in (-1, 1)$ - monotonic increasing, differentiable everywhere):
$$
t(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
ReLU ($\text{Domain} \in \mathbb{R} \text{; Range} \in [0, \infty)$ - monotonic non-decreasing, unbounded):
$$
r(x) = \max(0, x)
$$

![[noninear-activation.png]]