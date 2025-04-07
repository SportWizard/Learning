# What is mean square error?
Mean square error (MSE) is an object/loss function used in [[Machine Learning]] to calculate the loss between the actual value and the the prediction

# What problem requires MSE?
MSE is primary used in regression problem

# Equation
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \frac{1}{N} \|\hat{\vec{y}}-\vec{y}\|^2
$$
$\hat{y}_i \in \mathbb{R}$ is the prediction (e.g. $\vec{w}^\intercal \vec{x}_i$)
$y_i \in \mathbb{R}$ is the true value
$N$ is the total number of data points/samples

$\hat{\vec{y}} \in \mathbb{R}^m$ is vector containing predictions
$\vec{y}  \in \mathbb{R}^m$ s vector containing true values

# Difference between MAE and MSE
**MAE:** Linear loss - No matter the difference between the predicted value and the true value, MAE will output the same ratio for errors and punishments

**MSE:** Quadratic loss - The ratio for a small and a large difference between the predicted value and the true value is the ratio for errors and punishments is much close for small difference (e.g. 1 error : 2 punishments. While for large difference, the ratio for errors and punishments is much further apart (e.g. 100 errors : 100000 punishments)