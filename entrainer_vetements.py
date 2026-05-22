"""
Entraînement local des 2 modèles de classification de vêtements.

Produit :
  - modele_maison_vetements.keras       (CNN fait maison)
  - modele_transfer_vetements.keras     (MobileNetV2 transfer learning)

Adapté du notebook Colab pour fonctionner sur Windows + CPU.
"""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, applications, callbacks

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "images.csv"
IMAGE_DIR = BASE_DIR / "images_original"
LOG_DIR = BASE_DIR / "logs_marketplace"

CATEGORIES_UTILES = ["T-Shirt", "Shoes", "Pants", "Shirt", "Dress"]
NUM_CLASSES = len(CATEGORIES_UTILES)
TAILLE_ECHANTILLON = 4000
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# --- 1. Lecture du CSV ---
print("[1/7] Lecture du CSV de labels...")
df = pd.read_csv(CSV_PATH)
df["filename"] = df["image"].astype(str) + ".jpg"

# --- 2. Filtrage ---
print(f"[2/7] Filtrage sur {NUM_CLASSES} categories utiles...")
df_filtre = df[df["label"].isin(CATEGORIES_UTILES)].copy()
print(f"      Images correspondantes : {len(df_filtre)}")

# Vérifier que les fichiers existent
df_filtre["existe"] = df_filtre["filename"].apply(lambda f: (IMAGE_DIR / f).exists())
nb_manquants = (~df_filtre["existe"]).sum()
if nb_manquants:
    print(f"      [WARN] {nb_manquants} fichiers absents - on les ignore")
df_filtre = df_filtre[df_filtre["existe"]].drop(columns=["existe"])

if len(df_filtre) > TAILLE_ECHANTILLON:
    df_final = df_filtre.sample(n=TAILLE_ECHANTILLON, random_state=42)
else:
    df_final = df_filtre

print(f"      Echantillon final : {len(df_final)} images")
print(f"      Repartition :\n{df_final['label'].value_counts().to_string()}")

# --- 3. Pipelines de données ---
print("\n[3/7] Construction des pipelines TF...")
datagen = ImageDataGenerator(validation_split=0.2)

train_ds = datagen.flow_from_dataframe(
    dataframe=df_final,
    directory=str(IMAGE_DIR),
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=42,
)
val_ds = datagen.flow_from_dataframe(
    dataframe=df_final,
    directory=str(IMAGE_DIR),
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=42,
)
print(f"      Ordre des classes : {train_ds.class_indices}")

# --- 4. Callbacks ---
# (TensorBoard retire pour eviter la dependance tensorboard sur Windows.
#  Si tu en as besoin pour le rapport : pip install tensorboard puis remets-le.)
cb_maison = [
    callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
]
cb_transfer = [
    callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
]

# --- 5. Modèle 1 : CNN fait maison ---
print("\n[4/7] Construction CNN fait maison...")
inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

modele_maison = tf.keras.Model(inputs, outputs)
modele_maison.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
modele_maison.summary()

print("\n[5/7] Entrainement CNN maison...")
modele_maison.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb_maison, verbose=2
)

print("\nSauvegarde modele_maison_vetements.keras...")
try:
    modele_maison.save(BASE_DIR / "modele_maison_vetements.keras")
    print("[OK] Modele maison sauvegarde (.keras)")
except Exception as e:
    print(f"[WARN] Echec save .keras : {e}")
    modele_maison.save_weights(BASE_DIR / "modele_maison_vetements.weights.h5")
    print("[OK] Fallback : poids sauvegardes (.weights.h5)")

# --- 6. Modèle 2 : MobileNetV2 transfer ---
print("\n[6/7] Construction MobileNetV2 (transfer learning)...")
base_model = applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs_tl = tf.keras.Input(shape=(*IMG_SIZE, 3))
x_tl = applications.mobilenet_v2.preprocess_input(inputs_tl)
x_tl = base_model(x_tl, training=False)
x_tl = layers.GlobalAveragePooling2D()(x_tl)
x_tl = layers.Dropout(0.2)(x_tl)
outputs_tl = layers.Dense(NUM_CLASSES, activation="softmax")(x_tl)

modele_transfer = tf.keras.Model(inputs_tl, outputs_tl)
modele_transfer.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
modele_transfer.summary()

print("\n[7/7] Entrainement MobileNetV2...")
modele_transfer.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb_transfer, verbose=2
)

print("\nSauvegarde modele_transfer_vetements.keras...")
try:
    modele_transfer.save(BASE_DIR / "modele_transfer_vetements.keras")
    print("[OK] Modele MobileNetV2 sauvegarde (.keras)")
except Exception as e:
    print(f"[WARN] Echec save .keras : {e}")
    modele_transfer.save_weights(BASE_DIR / "modele_transfer_vetements.weights.h5")
    print("[OK] Fallback : poids sauvegardes (.weights.h5)")

print("\n=== ENTRAINEMENT TERMINE ===")
