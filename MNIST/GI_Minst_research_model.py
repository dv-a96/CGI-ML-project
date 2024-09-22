import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import mnist

# Load MNIST dataset
(x_train, y_train), _ =  tf.keras.datasets.mnist.load_data()
# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Flatten and normalize the input data
x_train_flat = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0

# Create reference matrix
num_of_measurement = 8
batch_size = 16
reference_matrix = tf.random.normal((num_of_measurement, 28 * 28))

# Multiply the reference matrix with the flattened input data
measurements = tf.matmul(reference_matrix, tf.transpose(x_train_flat))

# Reshape the data to fit the model input shape
x_train_reshaped = x_train.reshape(-1, 28, 28, 1).astype('float32')

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_reshaped, y_train, test_size=0.2, random_state=42)

# Define the model
neural_net = tf.keras.Sequential([
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),  # No need to specify input shape here
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(num_of_measurement * batch_size),  # Output layer with num_of_measurement * batch_size units
    tf.keras.layers.Reshape((batch_size, num_of_measurement)),  # Reshape layer to match the batch size and measurements
    tf.keras.layers.Conv1D(16, kernel_size=1, activation='relu'),  # 1D convolutional layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(32, kernel_size=1, activation='relu'),  # 1D convolutional layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10),  # Output layer with 10 units (for classification)
    tf.keras.layers.Activation('sigmoid')  # Sigmoid activation applied to the output
])

# Compile the model using an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
neural_net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Now build the model
neural_net.build(input_shape=(None, 28, 28, 1))

# Now you can access the summary
neural_net.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model_history = neural_net.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, epochs=5, callbacks=[early_stopping])

# Plot training history
plt.plot(model_history.history['accuracy'], label='Training Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
