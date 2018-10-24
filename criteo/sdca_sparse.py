import numpy as np
from scipy.sparse.linalg import norm
from math import exp

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
        alpha, omega = self.compute_updates(x, y*2 - 1.0, a_0, epochs, lamb)

        # averaging step, for non-smooth loss functions.
        # a = sum(alphas[t_0:])/(t-t_0 + 1)
        # w = sum(omegas[t_0:])/(t-t_0 + 1) # can also compute from a

        # for smooth loss functions, we just return the final value.
        a = alpha
        w = omega

        self.w = w
        self.a = a

        # print("w = " + str(w))
        return (w, a)

    # returns an array of predictions for each row datapoint of x
    def predict(self, X):
        values = X.dot(self.w)
        y = [(0 if val < 0 else 1) for val in values]
        return np.array(y)

    def getpvals(self, X):
        values = X.dot(self.w)
        pvals = [self.sigmoid(val[0,0]) for val in values]
        return pvals

    def compute_updates(self, x, y, a, epochs, lamb):
        # calculate initial omega from initial alpha. this is also sparse.
        w_0 = sum([x.getrow(i).multiply(a[i]) for i in range(len(a))]).multiply(1.0/(lamb*len(a))).transpose()

        # create lists of alpha and omega values
        # alphas = np.array([a])
        # omegas = np.array([w_0])

        # to save memory, we are only using the current value, not saving past values.
        # this is correct due to the use of a smooth loss function.
        alpha = a
        omega = w_0

        perm = np.array([np.arange(len(a))]).T

        # iterate and update, using epochs of random data permutations
        for k in range(epochs):
            np.random.shuffle(perm)

            # if k % 5 == 0:
            #     print(omega)

            for i in range(len(a)):
                updates = self.compute_update(int(perm[i]), x, y, alpha, omega, lamb)
                # alphas.append(updates[0])
                # omegas.append(updates[1])
                alpha = updates[0]
                omega = updates[1]

        return (alpha, omega)

    def compute_update(self, i, x, y, a, w, lamb):
        # copy a
        new_a = a.copy()
        new_w = w.copy()

        n = float(len(a))

        # compute gradient at point
        a_grad = self.compute_alpha_gradient(a[i], x.getrow(i), y[i], w, lamb, n)

        # use gradient to adjust the alpha and omega values
        new_a[i] += a_grad
        new_w += x.getrow(i).multiply(a_grad / (lamb * n)).transpose()
        return (new_a, new_w)

    # Log loss for use in logistic regression. This is a smooth loss function. Recommended by shalev-shwartz
    def compute_alpha_gradient_log(self, a, x, y, w, lamb, n):
        inside_term = 1.0 + exp(x.dot(w)[0,0] * y)
        numerator = y / inside_term - a
        denominator = max(1.0, 0.25 + (norm(x) ** 2)/(lamb * n))
        return numerator/denominator

    # Smooth hinge loss gradient closed form solution from pg 578 of shalev-shwartz paper.
    def compute_alpha_gradient_smooth_hinge(self, a, x, y, w, lamb, n, gamma=1.0):
        numerator = 1.0 - y * x.dot(w)[0,0] - gamma * a * y
        denominator = (norm(x)**2)/(lamb * n) + gamma #lamb*n in the denom
        value = numerator/denominator + a*y
        return y * max(0.0, min(1.0, value)) - a

    # Lipschitz hinge loss for SVM gradient closed form solution from pg 577 of shalev-shwartz paper.
    def compute_alpha_gradient_hinge(self, a, x, y, w, lamb, n):
        numerator = 1.0 - y * x.dot(w)[0,0]
        denominator = (norm(x)**2)/(lamb*n)
        value = numerator/denominator + a*y
        return y * max(0.0, min(1.0, value)) - a

    # sigmoid function to convert w.T.dot(x) into a probability for logistic regression
    # unused since the conversion isn't necessary, probably not needed but will leave here in case
    def sigmoid(self, z):
        probability = 1.0/(1.0 + exp(-1 * z))
        return probability
