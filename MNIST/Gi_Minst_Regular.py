from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print dataset shapes
print('X_train:', x_train.shape)
print('Y_train:', y_test.shape)
print('X_test: ', x_test.shape)
print('Y_test: ', y_test.shape)

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Display the pytorch image
plt.imshow(x_train[0], cmap='gray')
plt.show()

# Build the neural network model
neural_net = tf.keras.Sequential()
neural_net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
neural_net.add(tf.keras.layers.Dense(16, activation='relu'))
neural_net.add(tf.keras.layers.Dense(16, activation='relu'))
neural_net.add(tf.keras.layers.Dense(10, activation='softmax'))

# Display model summary
neural_net.summary()

# Compile the model
neural_net.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model_history = neural_net.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=10, callbacks=[early_stopping])

# Plot training history
plt.plot(model_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model - put your path
neural_net.save('C:\\Users\\shaha\\Desktop\\saved_model.h5')

# Load the saved model
loaded_model = tf.keras.models.load_model('C:\\Users\\shaha\\Desktop\\saved_model.h5')

# Test the loaded model using the testing set
test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)