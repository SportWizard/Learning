# What is linear discriminant analysis?
Linear discriminant analysis (LDA) is a concept and technique used in [[Machine Learning]] that implements linear dimension reduction on data samples

# Why dimension reduction?
The reason behind dimension reduction is to prevent over-fitting by reducing the number of dimension (a.k.a features), while keeping the most useful information

# Requirement
- Data
- Labels

# How does it work?
Binary class:
- Create an axis that maximizes the separation between the two class
- Project the data on to the new axis
More than two classes:
- Find a central between all classes
- Measure the distance between the mean of each class to the central
- Create at most $\text{number of classes} - 1$ axises that to the classes

# How is the axises created?
- Maximize the distance between the mean of classes
- Minimize the "scatter" (the distance between data of the same class)
- Apply $\frac{(\mu_1 - \mu_2)^2 + (\mu_1 - \mu_3)^2 + \cdots}{s_1^2 + s_2^2 + s_3^2 + \cdots}$

# Equation
LDA transformation:
- (Single data sample - vector):
$$
\vec{y} = A \vec{x}
$$
- (All data samples - matrix):
$$
Y = A X
$$
$\vec{x} \in \mathbb{R}^n$; $X \in \mathbb{R}^{n \times m}$ is the input(s)
$\vec{y} \in \mathbb{R}^n$; $Y \in \mathbb{R}^{n \times m}$ is the output(s)
$A \in \mathbb{R}^{m \times n}$ is the linear transformation

Note: A contains the top $k$ eigenvalues, where the top of the matrix captures the most information

![[lda.png]]

# Code
##### LDA with Scikit-learn
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2) # n_components is the number of principle components to keep

lda_X = lda.fit_transform(X, y)

# or
# lda.fit(X)
# lda_X = lda.transform(X)
```