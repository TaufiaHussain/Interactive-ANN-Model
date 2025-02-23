import tensorflow as tf
from tensorflow import keras
import numpy as np

# ✅ Load MNIST dataset from saved files
x_train = np.load("mnist_x_train.npy")
y_train = np.load("mnist_y_train.npy")
x_test = np.load("mnist_x_test.npy")
y_test = np.load("mnist_y_test.npy")

# ✅ Normalize pixel values (0 to 1 range)
x_train, x_test = x_train / 255.0, x_test / 255.0

# ✅ Define a better model with more neurons
inputs = keras.layers.Input(shape=(28, 28))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(128, activation='relu')(x)  # More neurons for better learning
x = keras.layers.Dense(64, activation='relu')(x)   # Added extra hidden layer
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# ✅ Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model with validation
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# ✅ Save the trained model
model.save("digit_recognizer.h5")
print("✅ Model training complete! Model saved as 'digit_recognizer.h5'")
