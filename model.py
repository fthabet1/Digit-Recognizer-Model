# Attempting to implement the model using a CNN with additional layers

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Loading the training dataset
trainData = pd.read_csv('train.csv')


# Separate the features and labels into two variables
Ytrain = trainData['label']
Xtrain = trainData.drop(columns = ['label'])

# Normalize the pixel values from (0, 255) to (0, 1)
Xtrain = Xtrain / 255.0

# Reshape the features to fit the CNN input shape
Xtrain = Xtrain.values.reshape(-1, 28, 28, 1) # Here, -1 means "infer the number of samples"

# Load the test dataset
testData = pd.read_csv('test.csv') # Test data has no labels
Xtest = testData / 255.0
Xtest = Xtest.values.reshape(-1, 28, 28, 1) # Reshape to 4D array for the CNN

# Building the CNN
model = Sequential([
    Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)), # First convolutional layer
    MaxPooling2D(pool_size = (2, 2)), # First pooling layer
    Conv2D(64, (3, 3), activation = "relu", input_shape = (28, 28, 1)), # Second convolutional layer
    MaxPooling2D(pool_size = (2, 2)), # Second pooling layer
    Flatten(),
    Dense(128, activation = "relu"), # Fully connected layer
    Dense(10, activation = "softmax") # Output layer for 10 classes
])


# Compile the model
model.compile(optimizer = "adam", loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Train the model
numEpochs = 25
history = model.fit(Xtrain, Ytrain, epochs = numEpochs, batch_size = 64, validation_split = 0.1)

# Predict on the test set
predictions = model.predict(Xtest)

# Convert predictions to class labels
predictedLabels = np.argmax(predictions, axis = 1)

for i in range(numEpochs):
    print(f"Accuracy of Epoch {i}: {history.history['accuracy'][i] * 100:.2f}%")

print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

# Prepare a submission file
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predictedLabels) + 1),
    "Label": predictedLabels
})

# Save the predictions ot a CSV file
submission.to_csv('submission.csv', index = False)
print("Submission file created: submission.csv")
