import sys
import numpy as np
from sklearn.svm import SVC
from src.utils import *
from src.linear_regression import *
from src.svm import *
from src.softmax import *
from src.features import *
from src.kernel import *


def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error

def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error

def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(
        train_x,
        train_y,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    # Test error mod 3
    # train_y_mod_3, test_y_mod_3 = update_y(train_y, test_y)
    # test_error_mod_3 = compute_test_error_mod3(test_x, test_y_mod_3, theta, temp_parameter)
    # return test_error_mod_3

    return test_error

def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    # YOUR CODE HERE
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y_mod_3, test_y_mod_3 = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(
        train_x,
        train_y_mod_3,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=3,
        num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y_mod_3, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_mod3.pkl.gz")

    return test_error


def run_softmax_pca_on_MNIST(temp_parameter=1, n_components=18):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function
    with dimension reduction using PCA

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """

    train_x, train_y, test_x, test_y = get_MNIST_data()

    # train_pca (and test_pca) is a representation of our training (and test) data
    # after projecting each example onto the first 18 principal components.
    ### Correction note:  the following 4 lines have been modified since release.
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_x_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_x_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

    theta, cost_function_history = softmax_regression(
        train_x_pca,
        train_y,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x_pca, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_pca.pkl.gz")

    return test_error


def plot_pca():
        n_components = 18
        train_x, train_y, test_x, test_y = get_MNIST_data()
        # train_pca (and test_pca) is a representation of our training (and test) data
        # after projecting each example onto the first 18 principal components.
        ### Correction note:  the following 4 lines have been modified since release.
        train_x_centered, feature_means = center_data(train_x)
        pcs = principal_components(train_x_centered)
        train_x_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
        plot_PC(
            train_x[range(000, 100),],
            pcs,
            train_y[range(000, 100)],
            feature_means) # feature_means added since release

        firstimage_reconstructed = reconstruct_PC(
            train_x_pca[0,],
            pcs,
            n_components,
            train_x,
            feature_means) # feature_means added since release
        plot_images(firstimage_reconstructed)
        plot_images(train_x[0, ])

        secondimage_reconstructed = reconstruct_PC(
            train_x_pca[1,],
            pcs,
            n_components,
            train_x,
            feature_means) # feature_means added since release
        plot_images(secondimage_reconstructed)
        plot_images(train_x[1, ])


def run_softmax_pca_on_MNIST(temp_parameter=1, n_components=10):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function
    with dimension reduction using PCA and a cubic kernel

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """

    train_x, train_y, test_x, test_y = get_MNIST_data()

    # train_pca (and test_pca) is a representation of our training (and test) data
    # after projecting each example onto the first 18 principal components.
    ### Correction note:  the following 4 lines have been modified since release.
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_x_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_x_pca = project_onto_PC(test_x, pcs, n_components, feature_means)
    # train_cube (and test_cube) is a representation of our training (and test) data
    # after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.
    train_cube = cubic_features(train_x_pca)
    test_cube = cubic_features(test_x_pca)

    theta, cost_function_history = softmax_regression(
        train_cube,
        train_y,
        temp_parameter,
        alpha=0.3,
        lambda_factor=1.0e-4,
        k=10,
        num_iterations=150)

    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_cube, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_pca.pkl.gz")

    return test_error


def run_cubic_svm_on_MNIST(n_components=10):
    train_x, train_y, test_x, test_y = get_MNIST_data()

    # train_pca (and test_pca) is a representation of our training (and test) data
    # after projecting each example onto the first 18 principal components.
    ### Correction note:  the following 4 lines have been modified since release.
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_x_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_x_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

    clf = SVC(random_state=0, kernel="poly", degree=3)    
    clf.fit(train_x_pca, train_y)

    predict_y = clf.predict(test_x_pca)
    test_error = 1 - np.mean(predict_y == test_y)
    return test_error


def run_rbf_svm_on_MNIST(n_components=10):
    train_x, train_y, test_x, test_y = get_MNIST_data()

    # train_pca (and test_pca) is a representation of our training (and test) data
    # after projecting each example onto the first 18 principal components.
    ### Correction note:  the following 4 lines have been modified since release.
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_x_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_x_pca = project_onto_PC(test_x, pcs, n_components, feature_means)
    
    clf = SVC(random_state=0, kernel="rbf")    
    clf.fit(train_x_pca, train_y)

    predict_y = clf.predict(test_x_pca)
    test_error = 1 - np.mean(predict_y == test_y)    
    return test_error


def display_usage():
    print("Usage: 'python3 main.py [option]")
    print("Options:")
    print("\t'plot_sample_data': Display the first few of the MNIST train set images")
    print("\t'linear_regression': Run the linear regression algorithm and display the test error")
    print("\t'svm_one_vs_rest': Run the SVM algorithm for classification of data in the One vs. Rest scheme and display the test error")
    print("\t'multi_svm': Run the Multi-class SVM algorithm for classification and report the test error")
    print("\t'softmax': Run the softmax classification algorithm and display the test errors for three temperature parameters")
    print("\t'softmax_mod3': Run the softmax classification with labels changed to their modulo 3 values and display the test error")
    print("\t'pca': Run the softmax classification with dimensionality reduction via PCA and display the test error")
    print("\t'plot_pca': Display the reconstruction of the first two train set data points after dimension reduction")
    print("\t'cubic': Run the softmax classification with the cubic kernel function and display the test error")
    print("\t'cubic_svm': Run the non-linear SVM algorithm using the cubic polynomial kernel and display the test error")
    print("\t'rbf_svm': Run the non-linear SVM algorithm using the RBF kernel and display the test error")
    print("Note: More options to come")


def main():
    #######################################################################
    # 1. Introduction
    #######################################################################
    try:
        if sys.argv[1] == "plot_sample_data":
            # Load MNIST data:
            train_x, _, _, _ = get_MNIST_data()
            print("Loaded MNIST data")
            # Plot the first 20 images of the training set.
            plot_images(train_x[0:20, :])

        #######################################################################
        # 2. Linear Regression with Closed Form Solution
        #######################################################################

        elif sys.argv[1] == "linear_regression":
            print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))
            print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.1))
            print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.01))

        #######################################################################
        # 3. Support Vector Machine
        #######################################################################
            
        elif sys.argv[1] == "svm_one_v_rest":
            print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())
        elif sys.argv[1] == "multi_svm":
            print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

        #######################################################################
        # 4. Multinomial (Softmax) Regression and Gradient Descent
        #######################################################################

        elif sys.argv[1] == "softmax":
            print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=0.5))
            print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))
            print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=2))

        #######################################################################
        # 6. Changing Labels
        #######################################################################

        elif sys.argv[1] == "softmax_mod3":
            print('softmax test_error_mod3=', run_softmax_on_MNIST_mod3(temp_parameter=1))

        #######################################################################
        # 7. Classification Using Manually Crafted Features
        #######################################################################

        ## Dimensionality reduction via PCA ##
        elif sys.argv[1] == "pca":
            print('softmax test_error=', run_softmax_pca_on_MNIST(temp_parameter=1))

        ## Plot PCA representation and data reconstruction ##
        elif sys.argv[1] == "plot_pca":
            plot_pca()

        ## Cubic Kernel ##
        elif sys.argv[1] == "cubic":
            print('softmax test_error=', run_softmax_pca_on_MNIST(temp_parameter=1))

        ## Cubic SVM ##
        elif sys.argv[1] == "cubic_svm":
            print("Cubic Polynomial SVM test_error=", run_cubic_svm_on_MNIST())

        ## RBF SVM ##
        elif sys.argv[1] == "rbf_svm":
            print("RBF SVM test_error=", run_rbf_svm_on_MNIST())

        # TODO: Implement RBF kernel softmax and add option to run it
            
        else:
            display_usage()
    
    except IndexError:
        display_usage()
            

if __name__ == "__main__":
    main()