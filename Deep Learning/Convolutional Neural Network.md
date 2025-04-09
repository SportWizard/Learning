# What is convolutional neural network?
Convolutional neural network (CNN) is an architecture in [[Deep Learning]] and it is widely used in computer vision and image processing. Some architecture that build on CNN are VGG, AlexNet, ResNet, UNet

![[cnn.png]]

# What problem is CNN used for?
CNN is used in image classification, segmentation and generation

# How does it work?
CNN uses a combination of convolutional layer, max-pooling to extract feature maps, it is then using as input to train the [[Fully Connected Neural Network]]

# Objective/loss function
[[Cross Entropy Error]] (other objective function can also be used)

# Spatial dimension
Spatial dimension are the axes that represent physical space (e.g., height, width, depth). These dimensions define how data is arranged in a geometric or grid-like layout, allowing algorithms to interpret spatial relationships

Example:
2D image has 
- Height (horizontal pixels of the image)
- Width (vertical pixels of the image)
- Channels (depth of the image)
	- Colour images has 3 channels (R: 0 - 255, G: 0-255, B:0-255)
	- Black and white images has 1 channel (0 - 255)

# Max-pooling
- Uses kernel (sliding window) (dimension depends on the spatial dimension of the input)
- Return maximum value of the kernel (sliding window) and the maximum value will represent every pixels of that window
- slide the window by $S$ pixels
- Repeat
##### Purpose:
- Used to reduce size of the layer
- Make the output less sensitive to small translation variations

##### Reduced layer size equation
$$
L_\text{height}' = \left\lfloor \frac{L_\text{height} + 2P_\text{height} - K_\text{height}}{S_\text{height}} \right\rfloor + 1
$$
$$
L_\text{width}' = \left\lfloor \frac{L_\text{width} + 2P_\text{width} - K_\text{width}}{S_\text{width}} \right\rfloor + 1
$$
$L_\text{height}' \in \mathbb{R}$ is the height of the output layer
$L_\text{height} \in \mathbb{R}$ is the height of the input layer
$P_\text{height} \in \mathbb{R}$ is the amount of padding added to $L_\text{height}$
$K_\text{height} \in \mathbb{R}$ is the height of the kernel
$S_\text{height} \in R$ is the number of stride (or pixels to slide the window) in the vertical direction

$L_\text{width}' \in \mathbb{R}$ is the width of the output layer
$L_\text{width} \in \mathbb{R}$ is the width of the input layer
$P_\text{width} \in \mathbb{R}$ is the amount of padding added to $L_\text{width}$
$K_\text{width} \in \mathbb{R}$ is the width of the kernel
$S_\text{width} \in \mathbb{R}$ is the number of stride (or pixels to slide the window) in the horizontal direction

# Batch normalization
Batch Normalization (BN) improves neural network training by normalizing the _activations_ of each layer using mini-batch statistics, then applying learned scaling and shifting parameters ($\gamma$, $\beta$)

##### Equations
$$
\vec{y} = BN_{\gamma, \beta}(\vec{x})
$$

- Normalize:
$$
\hat{x}_i = \frac{x_i - \mu_B(i)}{\sqrt{\sigma_B^2(i) + \epsilon}}
$$
$\hat{x}_i \in \mathbb{R}$ is an output of the normalization
$\mu_B^2(i) \in \mathbb{R}$; $\mu_B^2 = \frac{1}{m} \sum_{i=1}^m x_i$ is a mean
$\sigma_B^2(i) \in \mathbb{R}$; $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$ is a variance
$\epsilon \in \mathbb{R}$ (e.g. $\epsilon = 0.0001$), used to prevent division by 0 if $\sigma_B^2(i) = 0$

- Re-scale
$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$
$y_i \in \mathbb{R}$ is an output after re-scale
$\gamma_i \in \mathbb{R}$ scale the normalization
$\beta_i \in \mathbb{R}$ shifts the normalization

# Convolution
Convolution is a mathematical operation that extracts features such as horizontal lines, vertical lines, and diagonal lines using a sliding window called a kernel (or filter). After the convolution, a [[Nonlinear activations]] function (e.g. ReLU to remove all negative value) is applied to the output to introduce nonlinearity

##### Equations
$$
Y = \phi(\text{Conv(X)})
$$
$Y$ is the feature map(s)
$X$ is the input image
$\phi(\cdot)$ is the nonlinear activation function
$\text{Conv}(\cdot)$ is the convolutional

- Input vector, 1 channel, 1 kernel (output neurons/size: $L_\text{width}' = L_\text{width} - K_\text{width} + 1$ - but can be adjusted with zero-padding and striding):
$$
y_i = \sum_{j=1}^{K_\text{width}} w_j \times x_{i+j-1}; \; \forall i \in \{1, 2, \cdots, L'_\text{width}\}
$$
$y_i \in \mathbb{R}$ is an output of a pixel in the feature map; $\vec{y} \in \mathbb{R}^{L'_\text{width}}$ is the feature map
$w_j \in \mathbb{R}$; $\vec{w} \in \mathbb{R}^{K_\text{width}}$ is a weight in the kernel
$x_{i+j-1} \in \mathbb{R}$ is a pixel (or input) from the input layer; $\vec{x} \in \mathbb{R}^{L_\text{width}}$ is the input layer

$L_\text{width}' \in \mathbb{R}$ is the width of the feature map

$L_\text{width} \in \mathbb{R}$ is the width of the input image

$K_\text{width} \in \mathbb{R}$ is the width of the kernel

![[convolution1.png]]

- Input vector, multiple channels, 1 kernel:
$$
y_i = \sum_{j_2=1}^{L_\text{depth}} \sum_{j_1=1}^{K_\text{width}} w_{j_1, j_2} \times x_{i+j_1-1, j_2}; \; \forall i \in \{1, 2, \cdots, L'_\text{width}\}
$$
$y_i \in \mathbb{R}$ is an output of a pixel in the feature map; $\vec{y} \in \mathbb{R}^{L'_\text{width}}$ is the feature map
$w_{j, m} \in \mathbb{R}$; $\vec{w} \in \mathbb{R}^{K_\text{width} \times K_\text{depth}}$ is a weight in the kernel
$x_{i+j-1, m} \in \mathbb{R}$ is a pixel (or input) from the input layer; $\vec{x} \in \mathbb{R}^{L_\text{width} \times L_\text{depth}}$ is the input layer

$L_\text{width}' \in \mathbb{R}$ is the width of the feature map

$L_\text{width} \in \mathbb{R}$ is the width of the input image
$L_\text{depth} \in \mathbb{R}$ is the depth of the input image

$K_\text{width} \in \mathbb{R}$ is the width of the kernel
$K_\text{depth} \in \mathbb{R}$ is the depth of the kernel

Note: $L_\text{depth} = K_\text{depth}$

![[convolution2.png]]

- Input vector, multiple channels, multiple kernels:
$$
y_{i_1, i_2} = \sum_{j_2=1}^{L_\text{depth}} \sum_{j_1=1}^{K_\text{width}} w_{j_1, j_2, i_2} \times x_{i_1+j_1-1, j_2}; \; \forall i_1 \in \{1, 2, \cdots, L'_\text{width}\}; \; \forall i_2 \in \{1, 2, \cdots, k\}
$$
$y_{i_1, i_2} \in \mathbb{R}$ is an output of a pixel in the feature map; $Y \in \mathbb{R}^{L'_\text{width} \times k}$ is feature maps
$w_{j_1, j_2, i_2} \in \mathbb{R}$; $\vec{w} \in \mathbb{R}^{K_\text{width} \times K_\text{depth} \times k}$ is a weight in the kernel
$x_{i+j-1, j_2} \in \mathbb{R}$ is a pixel (or input) from the input layer; $\vec{x} \in \mathbb{R}^{L_\text{width} \times L_\text{depth}}$ is the input layer
$k \in \mathbb{R}$; $k \ge 1$ is the number of kernels

$L_\text{width}' \in \mathbb{R}$ is the width of the feature map

$L_\text{width} \in \mathbb{R}$ is the width of the input image
$L_\text{depth} \in \mathbb{R}$ is the depth of the input image

$K_\text{width} \in \mathbb{R}$ is the width of the kernel
$K_\text{depth} \in \mathbb{R}$ is the depth of the kernel

Note: $L_\text{depth} = K_\text{depth}$

![[convolution3.png]]

- Input matrix (2D image), multiple channels, multiple kernels:
$$
y_{i_1, i_2, i_3} = \sum_{j_3=1}^{L_\text{depth}} \sum_{j_2=1}^{L_\text{height}} \sum_{j_1=1}^{K_\text{width}} w_{j_1, j_2, j_3, i_3} \times x_{i_1+j_1-1, i_2+j_2-1, j_3}; \; \forall i_1 \in \{1, 2, \cdots, L'_\text{width}\}; \; \forall i_2 \in \{1, 2, \cdots, k\}
$$
$y_{i_1, i_2, i_3} \in \mathbb{R}$ is an output of a pixel in the feature map; $Y \in \mathbb{R}^{L'_\text{width} \times L'_\text{height} \times k}$ is feature maps
$w_{j_1, j_2, j_3, i_3} \in \mathbb{R}$ is a weight in the kernel; $\vec{w} \in \mathbb{R}^{K_\text{width} \times K_\text{height} \times K_\text{depth} \times k}$
$x_{i_1+j_1-1, i_2+j_2-1, j_3} \in \mathbb{R}$ is a pixel (or input) from the input layer; $\vec{x} \in \mathbb{R}^{L_\text{width} \times L_\text{height} \times L_\text{depth}}$ is the input layer
$k \in \mathbb{R}$; $k \ge 1$ is the number of kernels

$L_\text{width}' \in \mathbb{R}$ is the width of the feature map
$L_\text{height}' \in \mathbb{R}$ is the height of the feature map
$L_\text{depth}' \in \mathbb{R}$ is the depth of the feature map

$L_\text{width} \in \mathbb{R}$ is the width of the input image
$L_\text{height} \in \mathbb{R}$ is the height of the input image
$L_\text{depth} \in \mathbb{R}$ is the depth of the input image

$K_\text{width} \in \mathbb{R}$ is the width of the kernel
$K_\text{height} \in \mathbb{R}$ is the height of the kernel
$K_\text{depth} \in \mathbb{R}$ is the depth of the kernel

Note: $L_\text{depth} = L_\text{depth}' = K_\text{depth}$

![[convolution4.png]]

# Code
##### CNN using Tensorflow
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

# define the model structure using Keras
model = keras.models.Sequential([
	# Convolutional block
    keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same", input_shape=[28, 28, 1]), # [height, width, channels]
    keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(), # Flatten into a 1 dimensioal vector

	# FCNN
    keras.layers.Dense(units=7744, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax'),
])

# compile model by attaching loss/optimizer/metric components
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=3e-2), metrics=["accuracy"])

# learning a model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X, y_test))
```