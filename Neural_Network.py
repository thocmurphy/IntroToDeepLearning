# Call our standard library for typical imports/initializations
from StandardLib import *

# Import all necessary modules
from Model import Model
import Layer
import Activation
import Optimizer
import Loss
import Accuracy
import mnist_data

# Import and load MNIST data to train the model
training_path = mnist_data.import_fashion_mnist_dataset()
X, y, X_test, y_test = mnist_data.create_mnist_data(training_path)

# Shuffle the training data to train the model more effectively
X, y = mnist_data.shuffle_mnist_data(X, y)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer.Dense(X.shape[1], 128))
model.add(Activation.ReLU())
model.add(Layer.Dense(128, 128))
model.add(Activation.ReLU())
model.add(Layer.Dense(128, 10))
model.add(Activation.Softmax())

# Set loss, optimizer, and accuracy objects
model.set(
    loss=Loss.CategoricalCrossEntropy(),
    optimizer=Optimizer.Adam(decay=1e-3),
    accuracy=Accuracy.Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=128,
            print_every=100
            )

# Create an array with all images to test
image_data = ['tshirt.png', 'pants.png']

# Loop through each image
for image in image_data:
    # Get the path to the image
    path = f'testing_images/{image}'

    # Format the image to appear like an MNIST image
    data = mnist_data.format_images(path)
    
    # Predict on the image
    confidences = model.predict(data)

    # Get predictions
    predictions = model.output_layer_activation.predictions(confidences)

    # Get label from label index
    prediction = mnist_data.fashion_mnist_labels[predictions[0]]

    # Print the model's prediction
    print(prediction)