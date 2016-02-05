import numpy as np
import matplotlib.pyplot as plt
from src import LinearRegressionsMixture

# Generate N random input data points
N = 300
X = np.random.rand(N, 1)
tX = np.ones((N, 2))
tX[:, 1] = X[:, 0]

# Generate N random target values
w = np.random.rand(2, 2)
y = np.zeros(N)
n = int(np.random.rand(1, 1) * N)
y[:n] = np.dot(tX[:n, :], w[0, :]) + np.random.normal(size=n) * .03
y[n:] = np.dot(tX[n:, :], w[1, :]) + np.random.normal(size=N - n) * .01

# Plot the data set
rx = np.ones((100, 2))
r = np.arange(0, 1, .01)
rx[:, 1] = r
plt.plot(tX[:, 1], y, '.b')
plt.plot(r, np.dot(rx, w[0, :]), ':k', linewidth=2)
plt.plot(r, np.dot(rx, w[1, :]), ':k', linewidth=2)

# Model parameters
K = 2
epsilon = 1e-4
lam = 0.1
iterations = 50
random_restarts = 20

# Train the model
model = LinearRegressionsMixture(X, np.expand_dims(y, axis=1), K=K)
model.train(epsilon=epsilon, lam=lam, iterations=iterations, random_restarts=random_restarts, verbose=False)
print(model)

# Plot results
w1 = model.w[:2, 0]
w2 = model.w[:2, 1]
plt.plot(r, np.dot(rx, w1), '-r', label="1st component")
plt.plot(r, np.dot(rx, w2), '-g', label="2nd component")

# New point
x_new = [0.9]
y_new, y_posteriors = model.predict(x_new, posteriors=True)
plt.plot(x_new, y_new, 'kx', mew=4, ms=10, label="Prediction")
print("Posterior probabilities:")
print("p(z=1 | x_new=%s, y_new=%0.2f) = %0.4f" % (x_new[0], y_new, y_posteriors[0]))
print("p(z=2 | x_new=%s, y_new=%0.2f) = %0.4f" % (x_new[0], y_new, y_posteriors[1]))

plt.title("Mixture of two linear regressions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(numpoints=1, loc='best')
plt.savefig('result.pdf')