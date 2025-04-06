# What is autoencoder?
Autoencoder is an architecture and technique used in [[Machine Learning]] that implements non-linear dimension reduction on data samples using neural network

# How it works?
- Uses an encoder (created using [[Fully Connected Neural Network]]) to reduce the dimension of the data
- Uses a decoder (also created using [[Fully Connected Neural Network]]) to reconstruct the original data
- The reconstructed data will be used in the objective function to calculate its loss and adjust its free parameters (weights) using [[Backpropagation]]
- Once training is finish, the decoder will be abandoned since we only need the encoder to reduce the dimension of the data

# Objective/loss function
$$
\|\hat{X} - X\|
$$
$\hat{X}$: Reconstructed data
$X$: True data

![[autoencoder.png]]