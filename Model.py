from StandardLib import *
from Layer import Input
import Activation
import Loss
import pickle
import copy

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
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
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

    # Evaluates the model using passed-in dataset
    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default if batch size is not being set
        validation_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Make sure steps fit
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            # Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size : (step + 1)*batch_size]
                batch_y = y_val[step*batch_size : (step + 1)*batch_size]
            # Perform the forward pass
            output = self.forward(batch_X, training=False)
            # Calculate the loss
            self.loss.calculate(output, batch_y)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        # Print a summary
        print(f'validation, ' +
                f'acc: {validation_accuracy:.3f}, ' +
                f'loss: {validation_loss:.3f}')
    
    # Retrieves and returns the parameters of trainable layers
    def get_parameters(self):
        # Create a lost for parameters
        parameters = []
        # Iterate trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    
    # Updates the mode with new parameters
    def set_parameters(self, parameters):
        # Iterate over parameters and layers and update each layer with each set of parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file
    def save_parameters(self, path):
        # Open a file in the binary-write mode and save the parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with them
    def load_parameters(self, path):
        # Open a file in the binary-read more, load the weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):
        # Make a copy of the model
        model = copy.deepcopy(self)
        # Remove accumulated loss and accuracy
        model.loss.new_pass()
        model.accuracy.new_pass()
        # Remove any data in the input layer and reset gradients
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        # Iterate over all layers and remove their properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    # Loads and returns a model
    @staticmethod
    def load(path):
        # Open file in the binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    # Predicts based on the samples
    def predict(self, X, *, batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            # Make sure steps fit
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        # Model outputs
        output = []
        # Iterate over steps
        for step in range(prediction_steps):
            # If batch is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X
            else:
                batch_x = X[step*batch_X : (step + 1)*batch_X]
            # Perform the backwards pass
            batch_output = self.forward(batch_X, training=False)
            # Append batch predition to the list of predictions
            output.append(batch_output)
        # Stack and return results
        return np.vstack(output)

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
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)
        # If output activation is Softmax and loss function Categorical, create object
        if isinstance(self.layers[-1], Activation.Softmax) and isinstance(self.loss, Loss.CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation.Softmax_CategoricalCrossEntropy()
    
    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)
        # Default value if batch size is not being set
        train_steps = 1
        # If there is validation data passed, set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Ensure steps fit data
            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                # Ensure steps fit data
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        # Main training loop
        for epoch in range(1, epochs+1):
            # Print epoch number
            print(f'epoch: {epoch}')
            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            # Iterate over steps
            for step in range(train_steps):
                # If batch size is not set - train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                # Perform the forward pass
                output = self.forward(batch_X, training=True)
                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                # Perform backward pass
                self.backward(output, batch_y)
                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate:.3f}')
            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate:.3f}')
            # If there is the validation data
            if validation_data is not None:
                # Evaluate the model
                self.evaluate(*validation_data, batch_size=batch_size)