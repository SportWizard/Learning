# What is principle component analysis?
Principle component analysis (PCA) is a concept and technique used in [[Machine Learning]] that implements linear dimension reduction on data samples

# Why dimension reduction?
The reason behind dimension reduction is to prevent over-fitting by reducing the number of dimension (a.k.a features), while keeping the most useful information

# Requirement
- Data (no label require)

# How does it work?
- Create $n$ principles components (PC), where $n$ = number of dimension
- Retain the top $k$ PCs (sorted by eigenvalues) that capture the maximum variance, so data are as spread out as possible

# How are principle components determined?
The first PC is determined by maximizing the average distance between all the data projected on the current PC and the origin. The rest of the PCs also does the same but has to be orthogonal to all the existing PCs

# Equation
$$
\text{PCA transformation: } y = Ax
$$
$$
\text{Inverse PCA transformation: } x = A^\intercal y
$$
Note: A contains the top $k$ eigenvalues, where the top of the matrix captures the most information

![[pca.png]]
![[pca-chart.png]]

# Code
##### PCA with Scikit-learn
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # n_components is the number of principle components to keep

pca_X = pca.fit_transform(X)

# or
# pca.fit(X)
# pca_X = pca.transform(X)
```