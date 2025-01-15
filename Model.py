import StandardLib
from Layer import Input
from Activation import *
from Statistics import *

# Neural Network Model
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
    
    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss and optimizer
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    # Forward pass
    def forward(self, X, training):
        # Call forward method on input layer
        self.input_layer.forward(X, training)
        # Call forward method of every object in the chain
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    
    # Backward pass
    def backward(self, output, y):
        # If Softmax classifier
        if self.softmax_classifier_output is not None:
            # Call backward method
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            # Backward method on all other layers
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        # First call method on loss
        self.loss.backward(output, y)
        # Call over every layer in reverse
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Input()
        # Count all the objects
        layer_count = len(self.layers)
        # Initialize a list containing trainable layers
        self.trainable_layers = []
        # Iterate through
        for i in range(layer_count):
            # First layer is input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except the first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer (nect object is the loss)
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # Check for trainable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)
        # If output activation is Softmax and loss function Categorical, create object
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = Softmax_CategoricalCrossEntropy()
    
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)
        for epoch in range(1, epochs + 1):
            # Perform forward pass
            output = self.forward(X, training=True)
            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss
            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            # Perform backward pass
            self.backward(output, y)
            # Optimize
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            # print a summary
            if not epoch % print_every:
                print(
                    f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: { regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate:.3f}')
        if validation_data is not None:
            X_val, y_val = validation_data
            # Perform forward pass
            output = self.forward(X_val, training=False)
            # Calculate loss
            loss = self.loss.calculate(output, y_val)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            # Print a summary
            print(
                f'validation, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}'
            )