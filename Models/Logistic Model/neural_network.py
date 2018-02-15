"""
This file conatin a simple logistic model with neural network mindset
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

class LogisticNeuralModel:
    """
    This model finds whether it is a cat or not it uses
    Function ->
        + sigmoid -> This is helper function Compute the sigmoid of z
        + initialize_with_zeros -> This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
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

        A = 1 / (1 + np.exp(-z))

        return A

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros(shape=(dim, 1))
        b = 0

        return w, b


    def propagate(w, b, X, Y):
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
        
        Tips:
        - Write your code step by step for the propagation
        """
        pass

    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        pass

    def predict(w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        pass

    def model(self):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
        """
        w, b = self.initialize_with_zeros(5)
        print(w,b)

neuron = LogisticNeuralModel()

neuron.model()

