# What is K-nearest neighbours?
K-nearest neighbours (KNN) is a [[Supervised Learning]] algorithm used for classification and regression tasks. It makes prediction based on its k nearest neighbours

# How it works
**Training:**
	Map the inputs with the labels (this is already done when the data is given)
**Predict:**
	Find the k nearest neighbours of current data sample using metrics (Euclidean distance, Hamming distance, Manhattan distance), and classify it the same as the most occurring class

```python
import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbors=1):
        # Initalize attributes (instance variables)
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None

    def fit(self, X, y):
        # Store features with its labels
        self.X = X
        self.y = y

    def _euclidean_distances(self, X):
        # Since each test samples contains distances from it to all training sample, axis0 is the same size as X_test (a.k.a X) after broadcasting
        # Axis1 will contain the distances, which the same size as X_train (a.k.a self.X) after broadcasting
        # Axis2 will contain all the vectors
        train_X_reshape = self.X[np.newaxis, :, :] # Shape: (1, row, column)
        X_reshape = X[:, np.newaxis, :] # Shape: (row, 1, column)

        diff = train_X_reshape - X_reshape # NumPy will broadcasts train_X_reshape and X_reshape to the same shape
        squared_diff = np.square(diff) # Element-wise square
        sum_diff = np.sum(squared_diff, axis=2) # Summation of the differences between every X_test samples with all X_train samples
        return np.sqrt(sum_diff) # Element-wise square root

    def predict(self, X):
        # Raise an error if model was not fitted
        if self.X is None or self.y is None:
            raise ValueError("The KNNClassifier must be fitted using `fit(X, y)` before calling `predict`")

        distances = self._euclidean_distances(X)

        closest_samples_indexes = np.argsort(distances, axis=1)[:, :self.n_neighbors] # Select the closest n neighbors by sorting the distances per test sample (a.k.a row) and returning its index (The index is mapped to the original order)

        labels = self.y[closest_samples_indexes] # Get its corresponding label using the indexes

        predictions = np.array([np.bincount(label).argmax() for label in labels]) # Get the most occurring label for each test sample (a.k.a row). bincount() counts the number of occurrence of each number and stores it in array-like structure (For example, if 1 appear 3 times, then index 1 will have a 3). argmax() return the maximum value's index

        return predictions
```

```python
classifier = KNNClassifier(n_neighbors=k) # Create the classifier with k neighbors
classifier.fit(X_train, y_train) # Train kNN
y_pred = classifier.predict(X_test) # Predict on test set
```