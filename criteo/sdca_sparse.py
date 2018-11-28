import numpy as np
import math
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix

class SDCA:

    # loss_function can be {smooth_hinge, log, hinge}
    def __init__(self, loss_function='smooth_hinge'):
        if loss_function == 'smooth_hinge':
            self.compute_alpha_gradient = self.compute_alpha_gradient_smooth_hinge
        elif loss_function == 'log':
            self.compute_alpha_gradient = self.compute_alpha_gradient_log
        elif loss_function == 'hinge':
            self.compute_alpha_gradient = self.compute_alpha_gradient_hinge

    """performs SDCA on x with labels y for an SVM problem, with intial alpha a_0, epoch*n iterations.
    in this implementation, x is sparse, in the form of a scipy csr_matrix.
    a_0 is a transposed vector, lamb is a lambda regularization parameter, recommended to be 0.00001"""
    def train(self, x, y, a_0, epochs, lamb = 0.00001):
        # the optimal way to do this would be to first do a few normal SGD steps before using SDCA
        # for ease, I just directly went into SDCA
        # optimizing alpha step
        self.a, self.w = self.compute_updates(x, y*2 - 1.0, a_0, epochs, lamb)

        # averaging step, for non-smooth loss functions.
        # a = sum(alphas[t_0:])/(t-t_0 + 1)
        # w = sum(omegas[t_0:])/(t-t_0 + 1) # can also compute from a

        # for smooth loss functions, we just return the final value.

        return (self.w, self.a)

    # returns an array of predictions for each row datapoint of x
    def predict(self, X):
        values = X.multiply(self.w).sum(1)
        y = [(0 if val[0,0] < 0 else 1) for val in values]
        return np.array(y)

    def getpvals(self, X):
        values = X.multiply(self.w).sum(1)
        pvals = [self.sigmoid(val[0,0]) for val in values]
        return pvals

    def compute_updates(self, x, y, a, epochs, lamb):
        # calculate initial omega from initial alpha. this is also sparse.
        len_a = a.shape[1]
        scaling_factor = 1.0 / (lamb * len_a)
        w_0 = a.dot(x).multiply(scaling_factor)

        # create lists of alpha and omega values
        # alphas = np.array([a])
        # omegas = np.array([w_0])

        # to save memory, we are only using the current value, not saving past values.
        # this is correct due to the use of a smooth loss function.
        alpha = a
        omega = w_0

        perm = np.arange(len_a)

        # iterate and update, using epochs of random data permutations
        for k in range(epochs):
            np.random.shuffle(perm)

            for i in range(len_a):
                alpha, omega = self.compute_update(int(perm[i]), x, y, alpha, omega, lamb)

        return (alpha, omega)

    def compute_update(self, i, x, y, a, w, lamb):
        n = float(a.shape[1])
        xrow = x.getrow(i)
        # compute gradient at point
        a_grad = self.compute_alpha_gradient(a[0,i], xrow, y[i], w, lamb, n)

        # use gradient to adjust the alpha and omega values
        a[0,i] += a_grad
        w += xrow.multiply(a_grad / (lamb * n))
        return a, w

    # Log loss for use in logistic regression. This is a smooth loss function. Recommended by shalev-shwartz
    def compute_alpha_gradient_log(self, a, x, y, w, lamb, n):
        inside_term = 1.0 + math.exp(float(x.multiply(w).sum() * y))
        numerator = y / inside_term - a
        denominator = max(1.0, 0.25 + (norm(x) ** 2)/(lamb * n))
        return numerator/denominator

    # Smooth hinge loss gradient closed form solution from pg 578 of shalev-shwartz paper.
    def compute_alpha_gradient_smooth_hinge(self, a, x, y, w, lamb, n, gamma=1.0):
        numerator = 1.0 - y * x.multiply(w).sum() - gamma * a * y
        denominator = (norm(x)**2)/(lamb * n) + gamma
        value = numerator/denominator + a*y
        return y * max(0.0, min(1.0, value)) - a

    # Lipschitz hinge loss for SVM gradient closed form solution from pg 577 of shalev-shwartz paper.
    def compute_alpha_gradient_hinge(self, a, x, y, w, lamb, n):
        numerator = 1.0 - y * x.multiply(w).sum()
        denominator = (norm(x)**2)/(lamb*n)
        value = numerator/denominator + a*y
        return y * max(0.0, min(1.0, value)) - a

    # sigmoid function to convert w.T.dot(x) into a probability for logistic regression
    # unused since the conversion isn't necessary, probably not needed but will leave here in case
    def sigmoid(self, z):
        probability = 1.0/(1.0 + math.exp(-1 * z))
        return probability
