from StandardLib import *

'''
This file contains various Accuracy objects/functions to be used in a neural 
network.

All functions are based on the "Neural Networks from Scratch" textbook.
'''

# Common accuracy class
class Accuracy:
    # Calculate an accuracy
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    
    # Calculates accumulated accuracy
    def calculate_accumulated(self):
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Accuracy calculation for regression model
class Regression(Accuracy):
    def __init__(self):
        # Initialize a precision property
        self.precision = None
    
    # Calculates precision value based on passed-in information
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compares predictions to truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    

# Accuracy calculation for classification model
class Categorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary
    
    # Initialize
    def init(self, y):
        # No initialization is needed
        pass

    # Compares predictions to the truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y