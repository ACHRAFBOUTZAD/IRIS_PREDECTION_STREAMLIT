import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Charger le jeu de données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Définir les classes de CIFAR-10
class_names = ["Avion", "Automobile", "Oiseau", "Chat", "Cerf", "Chien", "Grenouille", "Cheval", "Bateau", "Camion"]

# Charger un modèle pré-entraîné (ici, MobileNetV2 pour une démo rapide)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Préparer l'interface Streamlit
st.title("Exploration et Classification du Jeu de Données CIFAR-10")

# Sélectionner une image aléatoire
image_index = st.slider("Choisissez un index d'image (0-9999)", 0, 9999, 0)
image = x_test[image_index]
label = y_test[image_index][0]

# Afficher l'image sélectionnée
st.write(f"Classe réelle : {class_names[label]}")
st.image(image, channels="RGB", use_column_width=True)

# Prédire la classe de l'image avec le modèle pré-entraîné
st.write("**Résultat de la Classification :**")

# Convertir l'image en RGB si elle est en niveaux de gris
if image.shape[-1] != 3:
    image = np.stack((image,)*3, axis=-1)

# Redimensionner et normaliser l'image pour MobileNetV2
resized_image = tf.image.resize(image, (224, 224)) / 255.0
input_image = np.expand_dims(resized_image, axis=0)

# Prédire avec MobileNetV2
predictions = model.predict(input_image)

# Afficher le résultat de la prédiction
top_5 = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
for i, (imagenet_id, label, score) in enumerate(top_5):
    st.write(f"{i+1}. {label}: {score*100:.2f}%")

# Afficher une matrice de quelques images du jeu de données CIFAR-10
st.write("### Aperçu du jeu de données CIFAR-10")
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(x_test))
    ax.imshow(x_test[idx])
    ax.set_title(class_names[y_test[idx][0]])
    ax.axis("off")
st.pyplot(fig)

st.write("Cette application utilise MobileNetV2 pour prédire la classe des images CIFAR-10. Vous pouvez expérimenter en choisissant différentes images avec le sélecteur ci-dessus.")
