# What is bias-variance tradeoff?
Bias-variance tradeoff is a concept in [[Machine Learning]] that describes the tradeoff between a simple and a complex model. The tradeoff is about balancing under-fitting (high bias) and over-fitting (high variance) to ensure the model **generalizes well** to unseen data.

# Equation
$$
\text{Average learning error} = \text{Bias}^2 + \text{Variance}
$$

# Bias
 - **Definition**: No matter how much the data changes, the model is the same, resulting in the very similar outputs
- **Cause:** Under-fitting
- ![[bias.png]]
# Variance
- **Definition:** The model changes when there is a slight change in the data, resulting in slightly different output each time
- **Cause:** Over-fitting
- ![[variance.png]]
- Variance tell us how spread out the data is. The higher the variance, the more spread out