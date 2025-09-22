Multi-Layer Neural Network

Overview

In this assignment, I built a multi-layer neural network from scratch using only NumPy.

The network uses sigmoid activation, mean squared error (MSE) as the loss function, and weights are updated using gradient descent with centered difference approximation.

The code is organized into small helper functions so each part of the workflow is clear and modular.

Functions and Their Roles

    initialize_weights(input, output, seed)

    Creates the weight matrix for each layer.
    The first column represents the bias weights.
    Using a random seed makes results reproducible.

2. Sigmoid(x)

    The activation function.
    Squashes values into a range between 0 and 1.

3. mse(y_true, y_pred)

    Calculates mean squared error.
    Used to evaluate how far predictions are from the desired outputs.

4. forward_pass(X, weight_matrix)

    Runs the input forward through all layers.
    Keeps a list of activations: starts with the input, then each hidden layer, and ends with the final output.

5. gradient(input, output, weight_matrix, layer_num, h)

    Approximates the gradient for one layer of weights using centered difference:

  (f(x+h)−f(x−h))/(2h)

    Perturbs each weight slightly, does a forward pass, and checks the change in loss.

6. backward_pass(input, output, weight_matrix, alpha, h)

    Computes gradients for all layers by calling gradient.
    Updates each weight matrix using gradient descent:

  W=W−α⋅∇W

7. evaluate(X, weights)

    Runs only the forward pass.
    Returns the final predictions of the network.
    Used for testing without modifying weights.

8. multi_layer_nn(...)

The main driver function that ties everything together.

Steps:

    Initializes all weight matrices.
    Loops over epochs.
    For each training sample:
    Calls backward_pass to update weights.
    After each epoch:
    Uses evaluate to get predictions on the test set.
    Computes MSE and stores it in an error list.
    After training ends:
    Runs one final evaluation on the test set.

Returns:

    Final trained weights,
    Error values after each epoch,
    Final predictions for the test set.

9. Workflow Explained (in simple terms)

At the start, the network has random weights.

During each epoch:

    Each training sample is passed in one at a time.
    The network predicts the output with a forward pass.
    The gradient is computed by nudging the weights slightly (using centered difference).
    The weights are updated using gradient descent.
    After all samples in an epoch, the network is tested on the test set and the error is recorded.
    The process repeats for the set number of epochs.
    Finally, the network outputs its learned weights, the error progression, and the test predictions.

10. Code Flow

Forward pass → Gradient calculation → Backward pass (update) → Test evaluation → Repeat for epochs.
