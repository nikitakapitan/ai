import numpy as np
from ai.target_functions import J_quadratic, J_quadratic_derivative
from ai.activation_functions import sigmoid, sigmoid_prime

def J(X, y):
    return J_quadratic(X, y)

def J_derivative(y, y_hat):
    return J_quadratic_derivative(y, y_hat)

def activation(x):
    return sigmoid(x)

def activation_derivative(x):
    return sigmoid_prime(x)




