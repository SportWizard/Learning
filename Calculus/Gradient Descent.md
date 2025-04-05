# What is gradient descent?
Gradient descent (GD) is concept that uses partial [[Derivative]] to find a minima

# How does it work?
Simplified version:
- GD uses partial derivatives to calculate the gradient of a function
- Uses the input in the gradient
- The output of the gradient is the implies whether the function is increasing or decreasing at that point and the direction of steepest ascent
- To convert gradient ascent into gradient descent, add a negative to flip the direction.

Example:
![[gradient-descent.png]]
As seen from the image, the gradient (or derivative) of $f(x) = x^2$ is $ f'(x)= x$, and the output of $f'(3) = 6$, suggesting that going in the positive direction (right) will be ascending. By adding a negative to the gradient, $-f'(3) = -6$, the output will be negative, which suggests that going in the negative direction will be descending.