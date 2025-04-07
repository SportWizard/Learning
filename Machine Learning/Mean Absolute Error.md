# What is mean absolute error?
Mean absolute error (MAE) is an object/loss function used in [[Machine Learning]] to calculate the loss between the actual value and the the prediction

# What problem requires MAE?
MAE is primary used in regression problem

# Equation
$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| \hat{y}_i - y_i \right|
$$
$\hat{y}_i \in \mathbb{R}$ is the prediction (e.g. $\vec{w}^\intercal \vec{x}_i$)
$y_i \in \mathbb{R}$ is the true value
$N$ is the total number of data points/samples

# Difference between MAE and MSE
**MAE:** Linear loss - No matter the difference between the predicted value and the true value, MAE will output the same ratio for errors and punishments

**MSE:** Quadratic loss - The ratio for a small and a large difference between the predicted value and the true value is the ratio for errors and punishments is much close for small difference (e.g. 1 error : 2 punishments. While for large difference, the ratio for errors and punishments is much further apart (e.g. 100 errors : 100000 punishments)