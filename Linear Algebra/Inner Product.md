# What is inner product?
Inner product builds off of [[Dot product]]

# Equation
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