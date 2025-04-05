# What is matrix multiplication?
Matrix multiplication is a type of [[Dot product]] for matrix. It allows you to map a set of vector from one dimension to another dimension

# Equation
Let $U_i$ and $V_i$ be column vectors
$$
\begin{align}
U \cdot V

&=

\begin{bmatrix}
u_{1 \, 1} & u_{1 \, 2} & \cdots & u_{1 \, m} \\
u_{2 \, 1} & u_{2 \, 2} & \cdots & u_{2 \, m} \\
\vdots & \vdots & \ddots & \vdots \\
u_{n \, 1} & u_{n \, 2} & \cdots & u_{n \, m} \\
\end{bmatrix}

\begin{bmatrix}
v_{1 \, 1} & v_{1 \, 2} & \cdots & v_{1 \, m} \\
v_{2 \, 1} & v_{2 \, 2} & \cdots & v_{2 \, m} \\
\vdots & \vdots & \ddots & \vdots \\
v_{n \, 1} & v_{n \, 2} & \cdots & v_{n \, m} \\
\end{bmatrix}^\intercal

\\
&=

\begin{bmatrix}
U_1 & U_2 & \cdots & U_m
\end{bmatrix}

\begin{bmatrix}
V_1 & V_2 & \cdots & V_m
\end{bmatrix}^\intercal

\\
&=

\begin{bmatrix}
U_1 & U_2 & \cdots & U_m
\end{bmatrix}

\begin{bmatrix}
V_1^\intercal \\ V_2^\intercal \\ \vdots \\ V_m^\intercal
\end{bmatrix}

\\
&=

U_1 \cdot V_1^\intercal + U_2 \cdot V_2^\intercal + \cdots + U_m \cdot V_m^\intercal
\end{align}
$$