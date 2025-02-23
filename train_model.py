import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# ✅ Use Functional API instead of Sequential to define input tensor
inputs = keras.layers.Input(shape=(28, 28))  # Explicit input tensor
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)  # Functional API model

# Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# ✅ Save Model Properly
model.save("digit_recognizer.h5")
print("✅ Model training complete! Model saved as 'digit_recognizer.h5'")
