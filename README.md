# Mixture of Linear Regressions
We propose an implementation of the mixtures of linear regression models, as described in Bishop [1]. The main changes are

 1. the precision parameter `beta` is learned independently for each component and not shared between all them
 2. the optional hyperparameter `lambda` that acts as a regularizer and might help in some cases.

## Usage
First, initialize a model of `K` components with a data set (`X`, `y`), where `X` is an `ndarray` of size (NxD) and `y` is an `ndarray` of size (Nx1):

````
model = LinearRegressionsMixture(X, y, K=2)
````

Then, train it with an initial value of beta:

````
model.train(epsilon=epsilon, lam=lam, iterations=iterations, random_restarts=random_restarts, verbose=False)
````

Finally, predict an output for a new input data point using:

````
y_new = model.predict(x_new)
````
**Note:** the predicted value is the **mean** of the predictive distribution.

You can optionally obtain the posterior probabilities of the new data point to belond to each of the components using:

````
y_new, y_posteriors = model.predict(x_new, posteriors=True)
```` 

A full working example can be found in `example.py`.