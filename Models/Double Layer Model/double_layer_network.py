"""
This file contain a double layer neural network
"""
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

class  DoubleLayerNeuralNetwork:
    """
    This model finds whether it is a cat or not it uses dual layer model
    Function ->
        + sigmoid -> This is helper function Compute the sigmoid of z
        + initialize_with_zeros -> This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        + initialize_with_rand -> This function creates a vector of random digits of shape (dim, 1) for w and initializes b to 0.
        + propagate -> Implement the cost function and its gradient for the propagation explained above
        + optimize -> This function optimizes w and b by running a gradient descent algorithm
        + predict -> Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        + model -> Builds the logistic regression model by calling the slave functions
    """

    def sigmoid(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        x -- A scalar or numpy array of any size.
        s = 1 / (1 + (e ^ -z) -> (Python np.exp(-z)))

        Return:
        s -- sigmoid(z)
        """

        s = 1 / (1 + np.exp(-z))

        return s

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)(A row vector)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros(shape=(dim, 1))
        b = 0

        return w, b

    def initialize_with_rand(self, nh_x, nh_y):
        """
        This function creates a vector of random of shape (dim, dim) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)(A row vector)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.random.rand(nh_x, nh_y)
        b = np.zeros(shape = (nh_y, 1) )

        return w, b

    def propagate(self, params, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
        m = X.shape[1]
        w1 = params['w1']
        w2 = params['w2']
        b1 = params['b1']
        b2 = params['b2']
        # Forward Propagation steps
        Z1 = np.dot(w1, X) + b1
        A1 = self.sigmoid(Z1)  # compute activation
        Z2 = np.dot(w2, A1) + b2
        A2 = self.sigmoid(Z2)  # compute activation
        # cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost

        # Backward Propgations steps
        dZ2 = A2 - Y
        dw2 = (1 / m) * np.dot(dZ2, A2.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(w2.T, dZ2) * (1 - np.tanh(Z1))
        dw1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        parameters = {
            "dw1" : dw1,
            "db1" : db1,
            "db2" : db2,
            "dw2" : dw2
        }
        return parameters

