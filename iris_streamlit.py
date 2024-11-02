import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import streamlit as st

# Load IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create and train a neural network model
def create_and_train_iris_model(num_neurons, epochs=50):
    # Build the model
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(4,)))  # Hidden layer with specified number of neurons
    model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=8, validation_data=(X_test, y_test), verbose=0)

    return model

# Train the model
model = create_and_train_iris_model(num_neurons=40, epochs=50)

# Streamlit app
st.title("Iris Flower Prediction App")

# Input fields for iris features
sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Button to make prediction
if st.button("Predict Iris Type"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    class_names = iris.target_names
    st.write(f"Predicted Iris Type: {class_names[predicted_class]}")
