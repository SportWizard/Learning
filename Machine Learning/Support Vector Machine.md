# What is support vector machine?
Support vector machine (SVM) is a linear and non-linear model in [[Machine Learning]] that uses [[Supervised Learning]] to train. It was once the state-of-the-art in many areas before the deep learning boom
# What problem is SVM used for?
SVM is primary used for binary classification problem, but can be used for multi-class classification with an addition method

# SVM terminologies
- **Decision boundary** is a line (in 2D) or hyperplane that separates the classes. The SVM tries to find the hyperplane that best divides the data into different classes
- **Support vectors** are the data points that are closest to the decision boundary (hyperplane) in the feature space, creating the margin boundaries. Support vectors are critical for defining the position of the hyperplane and thus the model itself.
- **Margin boundaries:** define the margin — the “safe zone” between the two classes.
- **Margin** is the distance between the decision boundary and the nearest support vector

- **Linearly separable** is there exists a decision boundary such that it can separate the class
- **Linearly non-separable** is there doesn't exists a decision boundary such that it can separate the class

# How does SVM work?
**Training:**
	 During training, it uses gradient descent to adjust its weight vector such that it achieve the maximum separation between the decision boundary and the support vectors
**Prediction:**
	SVM uses $\text{sign}(\vec{w}^\intercal \vec{x})$ to make its prediction

# What is special about SVM?
Since SVM adjusts the decision boundary that maximizes the margin, it is not affect by noise (can be thought of as small movement in data) that are not the support vectors. Meaning, all the other data points are irrelevant excepts for the support vectors

# Equation
$$
\hat{y} = \text{sign}(\vec{w}^\intercal \vec{x} + b)
$$
$\hat{y} \in \mathbb{R}$ is the output
$\vec{w} \in \mathbb{R}^n$ is a weight vector
$\vec{x} \in \mathbb{R}^n$ is an input vector
$b \in \mathbb{R}$ is the bias

# Objective/loss function
$$
\frac{\left| \vec{w}^\intercal \vec{x} + b \right|}{\|\vec{w}\|}
$$
This is the formula calculates the distance between a point to a line. In order to maximize the margin we would need to maximize this for every support vectors. We could make $\left| \vec{w}^\intercal \vec{x} + b \right| \to \infty$, but this is inefficient, rather we could minimize L2 norm in denominator, such that $\|\vec{w}\| \to 0$, but not exactly 0. Giving us this objective function

$$
\vec{w}^* = \arg \underset{\vec{w}, b}{\min} \|\vec{w}\|
$$
subject to
$$
y_i(\vec{w}^\intercal \vec{x}_i + b) \ge 1, \; \forall i \in \{1, 2, \cdots, N\}
$$

But this is more complicated and not as smooth to work with in optimization. so we convert to this
$$
\vec{w}^* = \arg \underset{\vec{w}}{\min} \frac{1}{2} \|\vec{w}\|^2 = \arg \underset{\vec{w}}{\min} \frac{1}{2} \vec{w}^\intercal \vec{w}
$$
subject to
$$
y_i(\vec{w}^\intercal \vec{x}_i + b) \ge 1, \; \forall i \in \{1, 2, \cdots, N\}
$$
___
That objective function only work if the data are linearly separable. To result this problem, we could allow some of the data point(s) to cross the decision boundary, introducing a bit of errors, giving us the final objective function for SVM
$$
\vec{w}^* = \arg \underset{\vec{w}, b, \xi_i}{\min} \|\vec{w}\|^2 + C \sum_{i=1}^N \xi_i = \arg \underset{\vec{w}}{\min} \frac{1}{2} \vec{w}^\intercal \vec{w} + C \sum_{i=1}^N \xi_i
$$
subject to
$$
y_i(\vec{w}^\intercal \vec{x}_i + b) \ge 1 - \xi_i, \; \forall i \in \{1, 2, \cdots, N\}
$$
$$
\xi_i \ge 0, \; \forall i \in \{1, 2, \cdots, N\}
$$
$\vec{w}^\intercal \vec{x}_i \in \mathbb{R}$ is the prediction
$y_i$ is the true value
$N$ is the total number of support vectors

$X = \begin{bmatrix}\vec{x}_1^\intercal \\ \vec{x}_2^\intercal \\ \vdots \\ \vec{x}_m^\intercal\end{bmatrix}\in \mathbb{R}^{m \times n}$ is a matrix containing $m$ inputs vectors with dimension/size $n$
$X \vec{w} \in \mathbb{R}^m$ is a vector containing prediction
$\vec{y} \in \mathbb{R}^m$ is vector containing true values

$C \ge 0$; $C \in \mathbb{R}$ is how much does the linear errors, $\xi_i$, contributes to the objective/loss function (0 mean it won't minimize the linear errors, which could cause poor performance)
$\xi_i \ge 0$; $\xi_i \in \mathbb{R}$ is the linear errors

- ## Hard margin SVM
	Hard margin SVM is an SVM such that it ensure there isn't any linear errors. This can be achieve by setting $C \approx \infty$

- ## Soft margin SVM
	Soft margin SVM is an SVM such that it allows some linear errors. This can be achieve by setting $C$ to a value that allows some linear errors

![[svm.png]]

# Nonlinear SVM
To extend SVM to nonlinear we can use **Kernels** to map data into higher-dimensional spaces where a hyperplane can more easily separate the data (In reality, it does computation that mimic mapping data into higher dimension, which is less computation expensive)

![[kernel-mapping.png]]

**Some popular kernel function:**
- Linear kernel (the default linear SVM)
- Polynomial kernel
- Gaussian (or RBF) kernel
	![[rbf-svm.png]]

# Multi-class SVMs
There are two way to do multi-class SVMs

- ## One-vs-one strategy
	**Training:**
	- ${K \choose 2} = \frac{K (K-1)}{2}$ SVMs is created, where $K$ is the number of classes
	- Each SVMs will be trained on a combination of 2 classes that hasn't been trained on
	**Prediction:**
	- Each SVMs will predict the input as either class, and using majority voting, it will classify the input as the class occurred the most from the output of all the SVMs (it doesn't matter if input doesn't belong either class in a SVM as it won't be the majority class)
	Example:
		Input is class B
		Training:
			SVM1 trained on A and B
			SVM2 trained on A and C
			SVM3 trained on B and C
		Prediction:
			SVM1 predicts B
			SVM2 predict A
			SVM3 predict B
			Majority voting: B
	![[one-vs-one.png]]

- ## One-vs-all strategy
	**Training:**
	- $K$ SVMs is created, $K$ is the number of classes
	- Each SVMs will be trained on a class and all other classes, but it will consider all the other classes as one class
	**Prediction:**
	- Each SVMs will predict the input as either class, and using A binary prediction (e.g., "yes/no" for being in that class) or a confidence score (like the distance from the hyperplane), it will classify the input as the class with the "yes" from the output of all the SVMs or the highest score
	Example:
		Input is class B
		Training:
			SVM1 trained on A and other classes as one class
			SVM2 trained on B and other classes as one class
			SVM3 trained on C and other classes as one class
		Prediction:
			SVM1 predicts No or 0.05
			SVM2 predict Yes or 0.85
			SVM3 predict No or 0.10
			Classify as: B
	![[one-vs-all.png]]

# Code
##### Linear SVM using Scikit-learn
```python
import numpy as np
from sklearn.svm import SVC

# C: determines the penalty for misclassifications or margin violations in the SVM objective function
linear_svm = SVC(kernel="linear", C=3) # Define model
linear_svm.fit(X, y) # Train model

predict = linear_svm.predict(X)
```

##### Nonlinear SVM using Scikit-learn
```python
import numpy as np
from sklearn.svm import SVC

# C: determines the penalty for misclassifications or margin violations in the SVM objective function
# Gamma: smoothness and complexity of the RBF kernel - high gamma: complex, wiggly decision boundaries, could cause overfitting, low gamma: smoother, simpler decision boundaries, could cause underfitting
rbf_svm = SVC(kernel="rbf", C=3, gamma=0.01) # Define model
rbf_svm.fit(X_train, y_train) # Train model

predict = rbf_svm.predict(X_train)
```