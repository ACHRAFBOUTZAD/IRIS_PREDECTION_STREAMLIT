import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Charger le modèle sauvegardé
model = tf.keras.models.load_model('iris_model.keras')

# Charger les données de l'iris
data = load_iris()
feature_names = data.feature_names
class_names = data.target_names

# Fonction pour faire des prédictions
def predict_iris_class(features):
    # Redimensionner les caractéristiques pour correspondre à l'entrée du modèle
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    class_index = np.argmax(prediction, axis=1)[0]
    return class_names[class_index]

# Interface utilisateur de l'application Streamlit
st.title("Prédiction des espèces de l'iris")
st.write("Veuillez entrer les caractéristiques suivantes de la fleur Iris :")

# Entrée des caractéristiques
sepal_length = st.slider('Longueur du sépale (cm)', min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.slider('Largeur du sépale (cm)', min_value=2.0, max_value=4.5, step=0.1)
petal_length = st.slider('Longueur du pétale (cm)', min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.slider('Largeur du pétale (cm)', min_value=0.1, max_value=2.5, step=0.1)

# Collecter les entrées utilisateur et prédire
features = [sepal_length, sepal_width, petal_length, petal_width]
if st.button("Prédire la classe de l'iris"):
    predicted_class = predict_iris_class(features)
    st.write(f'La classe prédite pour cette fleur est : **{predicted_class}**')
