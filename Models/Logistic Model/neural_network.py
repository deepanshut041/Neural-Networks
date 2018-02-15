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


    def propagate(self, w, b, X, Y):
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
        # Forward Propagation steps 
        A = self.sigmoid(np.dot(w.T, X) + b)  # compute activation
        cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost

        # Backward Propgations steps
        dz = A - Y
        dw = (1 / m) * np.dot(X, (dz).T)
        db = (1 /m ) * np.sum(dz)
        cost = np.squeeze(cost)
        parameters = {
            "dw" : dw,
            "db" : db,
            "cost" : cost
        }
        return parameters

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):
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
        costs = []
        for i in range(num_iterations):
            parameters = self.propagate(w, b, X, Y)
            dw = parameters['dw']
            db = parameters['db']
            cost = parameters['cost']

            # Appling gradient descent algorithm 
            w = w - learning_rate * dw
            b = b - learning_rate * db

            if i % 100 == 0:
                costs.append(cost)
                print("Cost after ", i, " iterations is - ", cost)
        
        params = {
            "w" : w,
            "b" : b
        }
        grads = {
            "dw": dw,
            "db": db
            }

        return params, grads, costs

    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
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
        w, b = self.initialize_with_zeros(X_train.shape[0])

        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

        Y_prediction_test = self.predict(parameters['w'], parameters['b'], X_test)
        Y_prediction_train = self.predict(parameters['w'], parameters['b'], X_train)
        
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        data = {
            "costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations
            }
        
        return data


# Reading data from data set
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_filename = os.path.join(BASE_DIR, 'datasets\\train_catvnoncat.h5')
test_filename = os.path.join(BASE_DIR, 'datasets\\test_catvnoncat.h5')

train_dataset = h5py.File(train_filename, 'r')
test_dataset = h5py.File(test_filename, 'r')

X_train_org = np.array(train_dataset["train_set_x"][:])
Y_train = np.array(train_dataset['train_set_y'][:])

X_test_org = np.array(test_dataset["test_set_x"][:])
Y_test = np.array(test_dataset['test_set_y'][:])

print("Training dataset shape - ", X_train_org.shape)
print("Test dataset shape - ", X_test_org.shape)

X_train_u = X_train_org.reshape(X_train_org.shape[0], -1).T
X_test_u = X_test_org.reshape(X_test_org.shape[0], -1).T

X_train = X_train_u / 255.
X_test = X_test_u / 255.

print("Flated Training dataset shape - ", X_train.shape)
print("Flated Testing dataset shape - ", X_test.shape)
neuron = LogisticNeuralModel()
data = neuron.model(X_train, Y_train, X_test, Y_test, 20000, 0.005)




