# README – Multi-Layer Neural Network (from Scratch)

This project implements a multi-layer neural network completely from scratch using NumPy.
The goal is to show how forward propagation, gradient calculation (using centered difference), and backpropagation can be built without relying on any deep learning libraries.

## 🚀 Functions Overview

### initialize_weights(input, output, seed)
- Initializes random weights for a layer.
- Includes bias in the first column.
- Uses a fixed seed to ensure reproducibility.

### sigmoid(x)
- The activation function used for all neurons.
- Squashes values into the range (0, 1).

### mse(y_true, y_pred)
- Calculates the Mean Squared Error (MSE).
- Used to measure how well the network is performing.

### forward_pass(X, weight_matrix)
- Performs forward propagation through the network.
- Returns a list of activations: starting with the input, then hidden layer outputs, and finally the network’s output.

### gradient(input, output, weight_matrix, layer_num, h)
- Calculates the gradient of weights for a specific layer using centered difference approximation:
  (f(x+h)−f(x−h))/(2h)
- Temporarily perturbs weights, runs forward pass, and computes the loss difference.

### backward_pass(input, output, weight_matrix, alpha, h)
- Performs backpropagation:
  - Calls gradient() for each layer.
  - Updates the weights using gradient descent:  
    W = W − α⋅dW

### evaluate(X, weights)
- Runs a forward pass on input X.
- Returns only the final output of the network (useful for testing).

### multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2)
- Main function that ties everything together.
- Trains the neural network and returns:
  - Final trained weight matrices
  - Error per epoch (MSE on test data)
  - Final output of the network on X_test

---

## 🔄 Workflow of the Code

1. **Initialize Weights**  
   Random weights (with bias) are created for each layer using initialize_weights.

2. **Training Loop (for each epoch):**
   - For every training sample:
     - Extract the sample (back_X, back_Y).
     - Run backward_pass:
       - Which calls gradient:
         - Which internally calls forward_pass.
       - Gradients are computed.
       - Weights are updated.
   - After all samples are processed → test the network on X_test:
     - Call evaluate to get predictions.
     - Compute MSE with mse.
     - Store the error for this epoch.

3. **After All Epochs:**
   - Do one final evaluation on X_test with updated weights.
   - Return weights, error list, and final predictions.

---

## 🧩 Example Flow (Simplified)

- Input sample goes through forward_pass.
- Loss is computed using mse.
- gradient perturbs weights to approximate partial derivatives.
- backward_pass uses gradients to update weights.
- After all samples in an epoch, evaluate runs on test data and stores error.
- Process repeats for all epochs.

This way, the network learns gradually, one sample at a time, and you can track its learning progress through the error values after each epoch.
