# Ramesh, Anirudh Kashyap
# 1002_216_351
# 2025_09_21
# Assignment_01_01

import numpy as np


def initialize_weights(input, output, seed):
    np.random.seed(seed)
    return np.random.randn(output, input + 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def forward_pass(X, weight_matrix):
    sig_list = [X]
    for i in range(len(weight_matrix)):
        z = np.dot(weight_matrix[i], np.vstack([np.ones((1, X.shape[1])), sig_list  [-1]]))
        sig_values = sigmoid(z)
        sig_list.append(sig_values)
    return sig_list

# Function to calculate gradients
def gradient(input, output, weight_matrix, layer_num, h):
    grad = np.zeros(weight_matrix[layer_num].shape)
    for i in range(weight_matrix[layer_num].shape[0]):
        for j in range(weight_matrix[layer_num].shape[1]):

            #math logic
            #temp=w+h

            weight_matrix[layer_num][i, j] += h
            activations_with_h = forward_pass(input, weight_matrix) 
            loss_with_h = mse(output, activations_with_h[-1])


            #Math logic
            # w=temp-h
            # temp2=w-h
            # So temp2 is equal to w-2h

            weight_matrix[layer_num][i, j] -= 2 * h
            activations_with_minus_h = forward_pass(input, weight_matrix)  
            loss_with_minus_h = mse(output, activations_with_minus_h[-1])
            weight_matrix[layer_num][i, j] += h

            # (f(x + h) - f(x - h))/2
            grad[i, j] = (loss_with_h - loss_with_minus_h) / (2 * h)  

    return grad

def backward_pass(input, output, weight_matrix, alpha, h):
    layer_num = len(weight_matrix)
    temp= [None] * layer_num

    for i in range(layer_num):
        temp[i] = gradient(input, output, weight_matrix, i, h)
    
    for i in range(layer_num):
        weight_matrix[i] = weight_matrix[i] - alpha * temp[i]
    return weight_matrix

# To evaluate the forward pass
def evaluate(X, weights):
    sig_list = forward_pass(X, weights)
    return sig_list[-1]


def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    number_of_input, number_of_train_samples = X_train.shape
    number_of_output, number_of_test_samples = Y_test.shape

    weights = []
    for i in range(len(layers)):
        if i == 0:
            input_size = number_of_input
        else:
            input_size = layers[i - 1]
        output_size = layers[i]

# Here with the help of the initialize_weights function I am  initializing the random weights for each layer
        weight_matrix = initialize_weights(input_size, output_size, seed)
        weights.append(weight_matrix)

    error_list = []
# Here I am initializing the output results with zeros
    res = np.zeros((number_of_output, number_of_test_samples))

#Outer loop iterates over the number of epochs
    for i in range(epochs):
        #Inner goes through all the samples 
        for j in range(number_of_train_samples):
            input = X_train[:, j:j + 1]
            output = Y_train[:, j:j + 1]




# #first backward pass is called as it calls gradient function and that function calls forward pass 
# So first the forward pass is carried out and gradients are calculated and backward pass updates the weights
# weight matrix is updated after every sample but after iterating over all samples the error is calculated
# And the same updated weights are used for next epoch

            weights = backward_pass(input, output, weights, alpha, h)

        pred_output = evaluate(X_test, weights)
        error = mse(Y_test, pred_output)
        error_list.append(error)


#With the final updated weights after all epochs the output is calculated with test set
    res = evaluate(X_test, weights)

    return weights, np.array(error_list), res
    pass