# What is inner product?
Inner product builds off of [[Dot product]] that always returns a scalar and a matrix for pair-wise inner product

# Equation
Transpose the vector/matrix such that the left one are row vector or matrix with row vectors and the right one are column vector or matrix with column vectors

##### Inner product
**Column vector**
$$
\begin{align}
\vec{v} \cdot \vec{u}

&=

\vec{v}^\intercal \vec{u}

\\
&=

\begin{bmatrix}
v_1 \\ v_2 \\ \vdots \\ v_n
\end{bmatrix}^\intercal

\begin{bmatrix}
u_1 \\ u_2 \\ \vdots \\ u_n
\end{bmatrix}

\\
&=

\begin{bmatrix}
v_1 & v_2 & \cdots & v_n
\end{bmatrix}

\begin{bmatrix}
u_1 \\ u_2 \\ \vdots \\ u_n
\end{bmatrix}

\\
&=

v_1 \cdot u_1 + v_2 \cdot u_2 + \cdots + v_n \cdot u_n

\\
&=

\sum_{i=1}^n v_i u_i
\end{align}
$$
$\vec{v} \in \mathbb{R}^n$
$\vec{u} \in \mathbb{R}^n$
$\vec{v}^\intercal \vec{u} \in \mathbb{R}$

**Row vector**
$$
\begin{align}
\vec{v} \cdot \vec{u}

&=

\vec{v} \vec{u}^\intercal

\\
&=

\begin{bmatrix}
v_1 & v_2 & \cdots & v_n
\end{bmatrix}

\begin{bmatrix}
u_1 & u_2 & \cdots & u_n
\end{bmatrix}^\intercal

\\
&=

\begin{bmatrix}
v_1 & v_2 & \cdots & v_n
\end{bmatrix}

\begin{bmatrix}
u_1 \\ u_2 \\ \vdots \\ u_n
\end{bmatrix}

\\
&=

v_1 \cdot u_1 + v_2 \cdot u_2 + \cdots + v_n \cdot u_n

\\
&=

\sum_{i=1}^n v_i u_i
\end{align}
$$
$\vec{v} \in \mathbb{R}^n$
$\vec{u} \in \mathbb{R}^n$
$\vec{v} \vec{u}^\intercal \in \mathbb{R}$

##### Pair-wise inner product
**Matrix with column vectors**
$$
\begin{align}
V \cdot U

&=

V^\intercal U

\\
&=

\begin{bmatrix}
\vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_m
\end{bmatrix}^\intercal

\begin{bmatrix}
\vec{u}_1 & \vec{u}_2 & \cdots & \vec{u}_m
\end{bmatrix}

\\
&=

\begin{bmatrix}
\vec{v}_1^\intercal \\ \vec{v}_2^\intercal \\ \vdots \\ \vec{v}_m^\intercal
\end{bmatrix}

\begin{bmatrix}
\vec{u}_1 & \vec{u}_2 & \cdots & \vec{u}_m
\end{bmatrix}

\\
&=

\begin{bmatrix}
\vec{v}_1^\intercal \vec{u}_1 & \vec{v}_1^\intercal \vec{u}_1 & \cdots & \vec{v}_1^\intercal \vec{u}_m
\\
\vec{v}_2^\intercal \vec{u}_1 & \vec{v}_2^\intercal \vec{u}_1 & \cdots & \vec{v}_2^\intercal \vec{u}_m
\\
\vdots & \vdots & \ddots & \vdots
\\
\vec{v}_m^\intercal \vec{u}_1 & \vec{v}_m^\intercal \vec{u}_1 & \cdots & \vec{v}_m^\intercal \vec{u}_m
\end{bmatrix}
\end{align}
$$
$V \in \mathbb{R}^{n \times m}$
$U \in \mathbb{R}^{n \times m}$
$V^\intercal U \in \mathbb{R}^{m \times m}$

**Matrix with row vectors**
$$
\begin{align}
V \cdot U

&=

V U^\intercal

\\
&=

\begin{bmatrix}
\vec{v}_1 \\ \vec{v}_2 \\ \vdots \\ \vec{v}_m
\end{bmatrix}

\begin{bmatrix}
\vec{u}_1 \\ \vec{u}_2 \\ \vdots \\ \vec{u}_m
\end{bmatrix}^\intercal

\\
&=

\begin{bmatrix}
\vec{v}_1^\intercal \\ \vec{v}_2^\intercal \\ \vdots \\ \vec{v}_m^\intercal
\end{bmatrix}

\begin{bmatrix}
\vec{u}_1 & \vec{u}_2 & \cdots & \vec{u}_m
\end{bmatrix}

\\
&=

\begin{bmatrix}
\vec{v}_1^\intercal \vec{u}_1 & \vec{v}_1^\intercal \vec{u}_1 & \cdots & \vec{v}_1^\intercal \vec{u}_m
\\
\vec{v}_2^\intercal \vec{u}_1 & \vec{v}_2^\intercal \vec{u}_1 & \cdots & \vec{v}_2^\intercal \vec{u}_m
\\
\vdots & \vdots & \ddots & \vdots
\\
\vec{v}_m^\intercal \vec{u}_1 & \vec{v}_m^\intercal \vec{u}_1 & \cdots & \vec{v}_m^\intercal \vec{u}_m
\end{bmatrix}
\end{align}
$$
$V \in \mathbb{R}^{m \times n}$
$U \in \mathbb{R}^{m \times n}$
$V U^\intercal \in \mathbb{R}^{m \times m}$

# Interesting concept
$$
\vec{v}^\intercal \vec{u} = \vec{u}^\intercal \vec{v}
$$
Proof:
$$
\begin{align}
\vec{v}^\intercal \vec{u}

&=

\begin{bmatrix}
v_1 \\ v_2 \\ \vdots \\ v_n
\end{bmatrix}^\intercal

\begin{bmatrix}
u_1 \\ u_2 \\ \vdots \\ u_n
\end{bmatrix}

\\
&=

\begin{bmatrix}
v_1 & v_2 & \cdots & v_n
\end{bmatrix}

\begin{bmatrix}
u_1 \\ u_2 \\ \vdots \\ u_n
\end{bmatrix}

\\
&=

v_1 \cdot u_1 + v_2 \cdot u_2 + \cdots + v_n \cdot u_n

\\
&=

u_1 \cdot v_1 + u_2 \cdot v_2 + \cdots + u_n \cdot v_n

\\
&=

\begin{bmatrix}
u_1 & u_2 & \cdots & u_n
\end{bmatrix}

\begin{bmatrix}
v_1 \\ v_2 \\ \vdots \\ v_n
\end{bmatrix}

\\
&=

\begin{bmatrix}
u_1 \\ u_2 \\ \vdots \\ u_n
\end{bmatrix}^\intercal

\begin{bmatrix}
v_1 \\ v_2 \\ \vdots \\ v_n
\end{bmatrix}

\\
&=

\vec{u}^\intercal \vec{v}
\end{align}
$$