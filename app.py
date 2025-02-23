import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_recognizer.h5")

# ✅ Load MNIST dataset
mnist_x_test = np.load("mnist_x_test.npy")
mnist_y_test = np.load("mnist_y_test.npy")

st.title("🧠 Interactive MNIST Digit Recognition")

# ✅ User Input for Selecting a Digit
selected_digit = st.number_input("Enter a digit (0-9) to test:", min_value=0, max_value=9, step=1)

# ✅ Find an Image of the Selected Digit
indices = np.where(mnist_y_test == selected_digit)[0]  # Get indices where the label matches input
if len(indices) > 0:
    test_index = np.random.choice(indices)  # Randomly pick an image of the chosen digit
    mnist_image = mnist_x_test[test_index]
    mnist_label = mnist_y_test[test_index]

    # ✅ Preprocess the image
    img_array = mnist_image / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Reshape for model input

    # ✅ Predict the digit
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # ✅ Display Prediction & Actual Label
    st.write(f"**Predicted Digit:** {predicted_digit}")
    st.write(f"**Actual Label:** {mnist_label}")

    # ✅ Display the Selected MNIST Image
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mnist_image, cmap="gray")
    ax.set_title(f"MNIST Image of Digit {selected_digit}")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("⚠ No images found for this digit in MNIST test set.")


