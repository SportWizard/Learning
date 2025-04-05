# What is derivative?
Derivative is concept in [[Calculus]] that describe the rate of change of a point.

# Another way to think about it
Since you can't calculate the slope of one point, the derivative can thought of as two points infinitely close to each to each other and you are calculating its slope

# Use cases
- $f'(x) = 0$ (horizontal slope) is used to find critical points, where it could be either a local maxima, minima or point of inflection
![[local-maxima-and-minima-points.png]]

# Equation
$$
\frac{d}{dx} f(x) = f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

# Product rule
$$
\frac{d}{dx} [f(x)g(x)] = \frac{df(x)}{dx} g(x) + f(x) \frac{dg(x)}{dx}
$$
$$
[f(x)g(x)]' = f'(x)g(x) + f(x)g'(x)
$$

# Quotient rule
$$
\frac{d}{dx} \frac{f(x)}{g(x)} = \frac{\frac{df(x)}{dx} g(x) - f(x) \frac{dg(x)}{dx}}{g(x)^2}
$$
$$
\left[\frac{f(x)}{g(x)}\right]' = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
$$

# Chain rule
$$
f(g(x))' = f'(g(x)) g(x)
$$