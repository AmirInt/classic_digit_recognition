import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each data point
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X: np.ndarray, theta: np.ndarray, temp_parameter: float):
    """
    Computes, for each data point X[i], the probability that X[i] is labelled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n data points each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labelled as j
    """
    #YOUR CODE HERE
    H = theta @ X.T / temp_parameter
    c = np.max(H, axis=0)
    H -= c
    H = np.exp(H)
    div = np.sum(H, axis=0)
    H /= div
    return H

def compute_cost_function(
        X: np.ndarray,
        Y: np.ndarray,
        theta: np.ndarray,
        lambda_factor: float,
        temp_parameter: float):
    """
    Computes the total cost over every data point.

    Args:
        X - (n, d) NumPy array (n data points each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    n = len(Y)
    H = compute_probabilities(X, theta, temp_parameter)
    H_log = np.log(np.choose(Y, H))
    sum = -np.sum(H_log) / n + lambda_factor / 2 * np.sum(theta * theta)
    return sum

def run_gradient_descent_iteration(
        X: np.ndarray,
        Y: np.ndarray,
        theta: np.ndarray,
        alpha: float,
        lambda_factor: float,
        temp_parameter: float):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n data points each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    k = theta.shape[0]
    n = len(Y)
    H = compute_probabilities(X, theta, temp_parameter)

    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n)).toarray()
    theta_g = -(M - H) @ X / (temp_parameter * n) + lambda_factor * theta
    theta -= alpha * theta_g

    # Alternative:
    # theta_g = np.zeros(theta.shape) # Theta gradient
    # for i in range(k):
    #     Y_i = (Y == i).astype(int)
    #     theta_g[i] = -np.sum(X * (Y_i - H[i]).reshape((len(Y), 1)), axis=0) / (temp_parameter * n) + lambda_factor * theta[i]
    
    # theta -= alpha * theta_g

    return theta


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each data point in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each data point in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each data point in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each data point in the test set
    """
    #YOUR CODE HERE
    return train_y % 3, test_y % 3

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n data points each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    return 1 - np.mean(Y == (get_classification(X, theta, temp_parameter) % 3))

def softmax_regression(
        X: np.ndarray,
        Y: np.ndarray,
        temp_parameter: float,
        alpha: float,
        lambda_factor: float,
        k: int,
        num_iterations: int):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for _ in range(num_iterations):
        cost_function_progression.append(
            compute_cost_function(
                X,
                Y,
                theta,
                lambda_factor,
                temp_parameter))
        theta = run_gradient_descent_iteration(
            X,
            Y,
            theta,
            alpha,
            lambda_factor,
            temp_parameter)
    return theta, cost_function_progression

def get_classification(X: np.ndarray, theta: np.ndarray, temp_parameter: float):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
