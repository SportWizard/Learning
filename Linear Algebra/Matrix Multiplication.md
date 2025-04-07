# What is matrix multiplication?
Matrix multiplication is a type of [[Dot product]] for matrix. It allows you to map a set of vector from one dimension to another dimension

# Equation
Let $\vec{v}_i$ and $\vec{u}_i$ be column vectors
$$
\begin{align}
V \cdot U

&=

\begin{bmatrix}
v_{1 \, 1} & v_{1 \, 2} & \cdots & v_{1 \, k} \\
v_{2 \, 1} & v_{2 \, 2} & \cdots & v_{2 \, k} \\
\vdots & \vdots & \ddots & \vdots \\
v_{n \, 1} & v_{n \, 2} & \cdots & v_{n \, k} \\
\end{bmatrix}

\begin{bmatrix}
u_{1 \, 1} & u_{1 \, 2} & \cdots & u_{1 \, m} \\
u_{2 \, 1} & u_{2 \, 2} & \cdots & u_{2 \, m} \\
\vdots & \vdots & \ddots & \vdots \\
u_{k \, 1} & u_{k \, 2} & \cdots & u_{k \, m} \\
\end{bmatrix}

\\
&=

\begin{bmatrix}
\sum_{i=1}^k v_{1 \, i} \times u_{i \, 1} & \sum_{i=1}^k v_{1 \, i} \times u_{i \, 2} & \cdots & \sum_{i=1}^k v_{1 \, i} \times u_{i \, m} \\
\sum_{i=1}^k v_{2 \, i} \times u_{i \, 1} & \sum_{i=1}^k v_{2 \, i} \times u_{i \, 2} & \cdots & \sum_{i=1}^k v_{2 \, i} \times u_{i \, m} \\
\vdots & \vdots & \ddots & \vdots \\
\sum_{i=1}^k v_{n \, i} \times u_{i \, 1} & \sum_{i=1}^k v_{n \, i} \times u_{i \, 2} & \cdots & \sum_{i=1}^k v_{n \, i} \times u_{i \, m} \\
\end{bmatrix}
\end{align}
$$
$V \in \mathbb{R}^{n \times k}$
$U \in \mathbb{R}^{k \times m}$
$V \cdot U \in \mathbb{R}^{n \times m}$

Note: $V$'s column must match $U$'s row