# What is under-fitting and over-fitting
Under-fitting and over-fitting are concepts of [[Machine Learning]] that describe how well the model captures the training data

# Under-fitting
- **Definition:** The model is too weak to capture any important information
- **Cause:** Model too simple
- Perform poorly on training and validation

- **Fix:** Increase model's complexity

- ![[under-fitting-regression.png]]
- ![[under-fitting-classification.png]]
# Over-fitting
- Definition: The model memorized the training data
- **Cause:** Model too complex

- Perform poorly on validation (unseen data)
- Captures more noises (unless information that can change the output of the modify with a slight change in the data)

- **Fix:** Decrease model's complexity, add more data, use regularizations

- ![[over-fitting-regression.png]]
- ![[over-fitting-classification.png]]