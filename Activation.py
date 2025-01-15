from StandardLib import *
from Statistics import CategoricalCrossEntropy

'''
This file contains various Activation objects/functions to be used in a neural 
network, including a combination of Softmax Activation and the Categorical Cross
Entropy Loss function for quicker backpropogation.

All classes and functions are based on the "Neural Networks from Scratch" textbook.
'''

# Rectified Linear Unit Activation
class ReLU:
    # Forward pass
    def forward(self, inputs, training):
        # Retain input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward Pass
    def backward(self, dvalues):
        # Need to modify original variable, make copy beforehand
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


# Sigmoid Activation
class Sigmoid:
    # Forward pass
    def forward(self, inputs, training):
        # Save inputs
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    # Backward Pass
    def backward(self, dvalues):
        # Since derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


# Softmax Activation
class Softmax:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobin matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Linear Activation
class Linear:
    # Forward pass
    def forward(self, inputs, training):
        # Remember values
        self.inputs = inputs
        self.output = inputs
    
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


# Softmax Classifier
class Softmax_CategoricalCrossEntropy():
    # Create activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true, training):
        # Output layer's activion function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If label are one-hot encoded, fix them
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate the gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize the gradient
        self.dinputs = self.dinputs / samples