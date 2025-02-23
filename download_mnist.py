from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save as NumPy files (optional)
import numpy as np
np.save("mnist_x_train.npy", x_train)
np.save("mnist_y_train.npy", y_train)
np.save("mnist_x_test.npy", x_test)
np.save("mnist_y_test.npy", y_test)

print("✅ MNIST dataset downloaded successfully!")
print(f"Training Data Shape: {x_train.shape}")  # (60000, 28, 28)
print(f"Testing Data Shape: {x_test.shape}")   # (10000, 28, 28)
