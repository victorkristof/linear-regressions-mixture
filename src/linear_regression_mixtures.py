from __future__ import print_function
import numpy as np
import scipy.linalg as lin
from .progressbar import ProgressBar
from .model import Model, ModelError
from scipy.stats import norm


class LinearRegressionsMixture(Model):
    """
    Mixture of linear regression models, as described in Bishop [1] (Section 14.5.1).

    [1] Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics).
    Springer-Verlag New York, Inc., Secaucus, NJ, USA, 2006.
    """

    # List of allowed keys for use with kwargs
    allowed_keys = ['K', 'beta', 'lam', 'iterations', 'epsilon', 'random_restarts']

    def __init__(self, X, y, K=2, lam=0, iterations=1000, epsilon=1e-4, random_restarts=None):
        """
        :param X:               Data set
        :param y:               Labels
        :param K:               Number of mixture components
        :param lam:             Regularization parameter for the regression weights
        :param iterations:      Maximum number of iterations
        :param epsilon:         Condition for convergence
        :param random_restarts: Restart the training with different random initial parameters
        """
        super(LinearRegressionsMixture, self).__init__(self.__class__.__name__ + " (%s components)" % K, X, y)

        # Model hyperparameters
        self.K = K
        self.lam = lam
        self.iterations = iterations
        self.epsilon = epsilon

        # Model parameters
        self.N, self.D = X.shape
        self.w = np.zeros((self.D, K))
        self.pi = np.zeros(K)
        self.gamma = np.zeros((self.N, K))
        self.beta = 1 / np.var(y)
        self.marginal_likelihood = - np.inf
        self.trained = False
        self.random_restarts = random_restarts

    def __str__(self):
        """
        Description of the model.

        :return: String representation of the model
        """
        description = "Model:        %s\n" % self.name
        if self.trained:
            description += "Likelihood:   %s\n" % self.marginal_likelihood
            description += "Beta:         %s\n" % self.beta
            description += "Lambda:       %s\n" % self.lam
            description += "Pi:           %s\n" % self.pi
            description += "Weights norm: %s" % [np.linalg.norm(self.w[:, i]) for i in range(self.K)]
            if self.w.shape[0] <= 20:
                description += "\n%s" % self.w
        else:
            description += "Not trained yet"
        return description

    def _expectation_maximization(self, X, y, verbose):
        """
        Learn the parameters for a mixture of least squares.
        Source:
        - http://stats.stackexchange.com/questions/33078/data-has-two-trends-how-to-extract-independent-trendlines/34287

        :return: Weights vectors, pi_k's, gamma's, beta and marginal likelihood
        """

        # Get the dimensions
        N = X.shape[0]
        D = X.shape[1] + 1  # + 1 to take bias into account

        if verbose:
            print("* Starting EM algorithm for mixture of K=%s least squares models" % self.K)
            print("* Beta = %s" % self.beta)
            print("* Lambda = %s" % self.lam)
            print("* Running at most %s iterations" % self.iterations)
            print("* Stopping when complete likelihood improves less than %s" % self.epsilon)

        # Add one's in order to find w0 (the bias)
        tX = np.concatenate((np.ones((N, 1)), X), axis=1)

        # Mixture weights
        pi = np.zeros(self.K) + 1/float(self.K)

        # Expected mixture weights for each data point (responsibilities)
        gamma = np.zeros((N, self.K)) + .5

        # Regression weights
        w = np.random.rand(D, self.K)

        # Precision parameter, initialized to the reciprocal of the true variance of the targets
        beta = np.zeros(self.K) + self.beta

        # Initialize likelihood
        complete_log_likelihood = - np.inf
        complete_log_likelihood_old = - np.inf

        if verbose:
            print("Obj\tpi1\tpi2\tw11\tw12\tw21\tw22\tbeta1\tbeta2")

        for i in range(self.iterations):

            # E-step

            # Compute Likelihood for each data point
            err = (np.tile(y, (1, self.K)) - np.dot(tX, w)) ** 2  # y - <w_k, x_n>
            prbs = - 0.5 * np.tile(beta, (N, 1)) * err
            probabilities =  np.tile(np.sqrt(beta), (N, 1)) * np.exp(prbs)  # N(y_n | <w_k, x_n>, beta^{-1})

            # Compute expected mixture weights
            gamma = np.tile(pi, (N, 1)) * probabilities
            gamma /= np.tile(np.sum(gamma, 1), (self.K, 1)).T

            # M-step

            # Max with respect to the mixture probabilities
            pi = np.mean(gamma, axis=0)

            # Max with respect to the regression weights
            for k in range(self.K):
                R_k = np.diag(gamma[:, k])
                R_kX = R_k.dot(tX)
                L = R_kX.T.dot(tX) + np.eye(D) * self.lam  # also try: lam / beta
                R = R_kX.T.dot(y)
                w[:, k] = lin.solve(L, R)[:, 0]

            # Max with respect to the precision term
            beta = N * pi / np.sum(gamma * err)

            # Evaluate the complete data log-likelihood to test for convergence
            complete_log_likelihood = float(np.sum(np.log(np.sum(np.tile(pi, (N, 1)) * probabilities, axis=1))))

            if verbose:
                print("%0.2f\t\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f" % (complete_log_likelihood,
                                                                                  pi[0], pi[1],
                                                                                  w[0, 0], w[1, 0],
                                                                                  w[0, 1], w[1, 1],
                                                                                  beta[0], beta[1]))
            # if np.isnan(complete_log_likelihood) \
            #             or np.abs(complete_log_likelihood - complete_log_likelihood_old) < self.epsilon:
            #     return w, pi, gamma, beta, complete_log_likelihood
            try:
                if np.isnan(complete_log_likelihood) \
                        or complete_log_likelihood < complete_log_likelihood_old \
                        or np.abs(complete_log_likelihood - complete_log_likelihood_old) < self.epsilon:
                    return w, pi, gamma, beta, complete_log_likelihood
            except:
                pass

            complete_log_likelihood_old = complete_log_likelihood

        # Hitting maximum number of iterations
        return w, pi, gamma, beta, complete_log_likelihood

    def _train(self, X, y, verbose, silent):
        """
        Private method to train the model.

        :param X:       Training set
        :param y:       Training labels
        :param verbose: Display details during expectation maximization
        :param silent:  Hide the progress bar
        :return:        Parameters of the model and data likelihood
        """
        if self.random_restarts and self.random_restarts > 0:
            w = pi = gamma = beta = None
            marginal_likelihood = - np.inf
            if not silent:
                if not verbose:
                    bar = ProgressBar(self.random_restarts, count=True, text="Random restarts")
                    bar.start()
            for r in range(self.random_restarts):
                w_new, pi_new, gamma_new, beta_new, marginal_likelihood_new = self._expectation_maximization(X, y, verbose)
                if marginal_likelihood_new > marginal_likelihood:
                    # print("Improved solution!")
                    w = w_new.copy()
                    pi = pi_new.copy()
                    gamma = gamma_new.copy()
                    beta = beta_new
                    marginal_likelihood = marginal_likelihood_new
                if not silent:
                    if not verbose:
                        bar.update(r)
        else:
            w, pi, gamma, beta, marginal_likelihood = self._expectation_maximization(X, y, verbose)

        return w, pi, gamma, beta, marginal_likelihood

    def train(self, seed=None, verbose=False, silent=False, **kwargs):
        """
        Train a mixture of least squares.

        :param seed:    Set the seed to fix the randomness
        :param verbose: Display details during expectation maximization
        :param silent:  Force to display nothing (a progress bar is displayed with random restarts even if non verbose)
        :param kwargs:  Set some hyperparameters if needed
        """
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)
        if seed:
            np.random.seed(seed)
        self.w, self.pi, self.gamma, self.beta, self.marginal_likelihood = self._train(self.X, self.y, verbose, silent)
        self.trained = True

    def _cross_validate(self, k_fold, verbose, silent):
        """
        Private method to perform k-fold cross-validation.

        :param k_fold:  Number of folds
        :param verbose: Display state of evaluation
        :param silent:  Hide the progress bar
        :return:        Mean RMSE, std dev RMSE
        """
        rmse_all = []
        if not silent:
            count = 0
            bar = ProgressBar(k_fold, count=True, text="Cross-Validation")
            bar.start()
        for X_train, y_train, X_valid, y_valid in self._k_fold_generator(k_fold):
            rmse = self._evaluate(X_train, X_valid, y_valid, verbose)
            rmse_all.append(rmse)
            if not silent:
                bar.update(count)
                count += 1
        return np.mean(rmse_all), np.std(rmse_all)

    def cross_validate(self, k_fold=10, verbose=False, silent=False):
        """
        Perform k-fold cross-validation on the trained model.

        :param k_fold:  Number of folds
        :param verbose: Display state of evaluation
        :param silent:  Hide the progress bar
        :return:        Mean RMSE, std dev RMSE
        """
        if k_fold <= 1:
            raise(ModelError("Parameter k_fold must at least be equal to 2"))
        if self.trained:
            rmse_avg, rmse_std = self._cross_validate(k_fold, verbose, silent)
            return rmse_avg, rmse_std
        else:
            raise(ModelError("Model not trained"))

    @staticmethod
    def _find_hyperparameters(lambdas, scores, verbose):
        """
        Extract the optimal hyperparameter according to some scores.

        :param lambdas: Ndarray of lambdas
        :param scores:  Ndarray of scores
        :param verbose: Display the optimal parameters
        :return:        Optimal beta, optimal lambda
        """
        indices = np.unravel_index(np.argmin(scores), np.shape(scores))
        optimal_lambda = lambdas[indices[0]]
        if verbose:
            print("Optimal lambda: %s" % optimal_lambda)
        return optimal_lambda

    def grid_search(self, lambdas, k_fold=1, X_test=None, y_test=None, verbose=False, silent=False):
        """
        Train the model using grid search over lambda.

        :param lambdas: Ndarray of lambdas
        :param k_fold:  If > 1, cross-validation is used for each combination using self.X and self.y. Otherwise must pass X_test and y_test
        :param X_test:  Test set if cross-validation is not used
        :param y_test:  Test label if cros-validation is not used
        :param verbose: Display details during training
        :param silent:  Hide the progress bar
        :return:        RMSE for each combination (avg and std dev if cross-validated)
        """
        rmse_all = []
        rmse_std_all = []
        if not silent:
            count = 0
            # +1 to account for the final training
            bar = ProgressBar(len(lambdas) + 1, text="Grid Search", count=True)
            bar.start()
        for i, l in enumerate(lambdas):
            self.lam = l
            rmse_i = []
            rmse_std_i = []
            self.w, self.pi, self.gamma, self.beta, self.marginal_likelihood = self._train(self.X, self.y, verbose, silent=True)
            if k_fold == 1:
                if X_test and y_test:
                    rmse = self.evaluate(X_test, y_test)
                    rmse_i.append(rmse)
                else:
                    raise(ModelError("Testing set not valid"))
            else:
                rmse_avg, rmse_std = self._cross_validate(k_fold, verbose, silent=True)
                rmse_i.append(rmse_avg)
                rmse_std_i.append(rmse_std)
            if not silent:
                bar.update(count)
                count += 1

            rmse_all.append(rmse_i)
            if k_fold > 1:
                rmse_std_all.append(rmse_std_i)

        # Extract the best hyperparameters
        self.lam = self._find_hyperparameters(lambdas, rmse_all, verbose=verbose)
        # Train the final model
        self.w, self.pi, self.gamma, self.beta, self.marginal_likelihood = self._train(self.X, self.y, verbose=verbose, silent=True)
        if not silent:
            bar.update(count)
            count += 1
        self.trained = True

        if k_fold == 1:
            return rmse_all
        else:
            return rmse_all, rmse_std_all

    def _predict(self, x_new, posteriors):
        """
        Private method to predict the value of x_new.

        :param x_new:       New data point (D)
        :param posteriors:  Whether or not returning the predictions together with the posterior probabilities
        :return:            Predicted value and index of corresponding mixture component
        """
        tx = np.ones((1, len(x_new ) + 1))
        tx[0, 1:] = x_new
        y_new = 0

        for k in range(self.K):
            w_k = self.w[:, k]
            pi_k = self.pi[k]
            y_new += pi_k * np.dot(tx, w_k)[0]

        if posteriors:
            y_posteriors = []
            for k in range(self.K):
                pi_k = self.pi[k]
                beta_k = self.beta[k]
                w_k = self.w[:, k]
                mu = np.dot(tx, w_k)[0]
                sigma = 1 / np.sqrt(beta_k)
                y_posteriors.append(pi_k * norm.pdf(y_new, loc=mu, scale=sigma))
            y_posteriors = np.array(y_posteriors)
            normalization = y_posteriors.sum()
            y_posteriors = y_posteriors / normalization
            return y_new, y_posteriors
        else:
            return y_new


    def predict(self, x_new, posteriors=False):
        """
        Predict the value of a new data point. To do so, it finds the closest training point and uses the parameters of
        the most likely component. Optionally return every prediction with the corresponding posterior probabilities.

        :param x_new:       New data point (size D)
        :param posteriors:  Whether or not returning the predictions together with the posterior probabilities
        :return:            Predicted value and index of corresponding component or predicted values and posteriors
        """
        if self.trained:
            x_new = list(x_new)
            if len(x_new) == self.X.shape[1]:
                return self._predict(x_new, posteriors)
            else:
                raise(ModelError("Invalid size for new data point (%s instead of %s)" % (len(x_new), self.X.shape[1])))
        else:
            raise(ModelError("Model not trained"))

    def _evaluate(self, X_test, y_test, verbose):
        """
        Private method to evaluate the model against a test set.

        :param X_test:  Test set (N* x D)
        :param y_test:  Test labels (N*)
        :param verbose: Display details during the evaluation
        :return:        Total RMSE, accuracy and a counter of chosen mixture components
        """
        se_total = []
        if verbose:
            bar = ProgressBar(end_value=X_test.shape[0], text="Data point", count=True)
            bar.start()
        for i, x_new in enumerate(X_test):
            y_actual = y_test[i][0]
            y_new = self._predict(list(x_new), posteriors=False)
            se = (y_actual - y_new)**2
            se_total.append(se)
            if verbose:
                bar.update(i)

        rmse = np.sqrt(np.mean(se_total))

        if verbose:
            print("RMSE: %s" % rmse)

        return rmse

    def evaluate(self, X_test, y_test, verbose=False):
        """
        Evaluate the model against a test set.

        :param X_test:  Test set (N* x D)
        :param y_test:  Test labels (N*)
        :param verbose: Display details during the evaluation
        :return:        Total RMSE
        """
        if verbose:
            print("Evaluating model %s..." % self.name)
        return self._evaluate(self.X, X_test, y_test, verbose)
