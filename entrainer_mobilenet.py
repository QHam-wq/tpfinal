"""
Entraîne UNIQUEMENT le modèle MobileNetV2 (transfer learning).
Contournement SSL : Windows + certificats CA mal configurés bloquent le
téléchargement des poids ImageNet depuis storage.googleapis.com.
"""

import os
import sys
import ssl
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

# Contournement SSL pour le téléchargement des poids ImageNet
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, applications, callbacks

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "images.csv"
IMAGE_DIR = BASE_DIR / "images_original"

CATEGORIES_UTILES = ["T-Shirt", "Shoes", "Pants", "Shirt", "Dress"]
NUM_CLASSES = len(CATEGORIES_UTILES)
TAILLE_ECHANTILLON = 4000
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# --- Préparation du dataset (identique au script principal) ---
print("[1/4] Lecture et filtrage du CSV...")
df = pd.read_csv(CSV_PATH)
df["filename"] = df["image"].astype(str) + ".jpg"
df = df[df["label"].isin(CATEGORIES_UTILES)].copy()
df["existe"] = df["filename"].apply(lambda f: (IMAGE_DIR / f).exists())
df = df[df["existe"]].drop(columns=["existe"])
if len(df) > TAILLE_ECHANTILLON:
    df = df.sample(n=TAILLE_ECHANTILLON, random_state=42)
print(f"      {len(df)} images selectionnees")

print("[2/4] Construction pipelines TF...")
datagen = ImageDataGenerator(validation_split=0.2)
train_ds = datagen.flow_from_dataframe(
    df, directory=str(IMAGE_DIR), x_col="filename", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training", seed=42,
)
val_ds = datagen.flow_from_dataframe(
    df, directory=str(IMAGE_DIR), x_col="filename", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation", seed=42,
)

print("[3/4] Telechargement et construction MobileNetV2...")
base_model = applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

modele = tf.keras.Model(inputs, outputs)
modele.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
modele.summary()

print(f"[4/4] Entrainement MobileNetV2 ({EPOCHS} epochs max, EarlyStopping)...")
cb = [callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
modele.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, verbose=2)

print("\nSauvegarde modele_transfer_vetements.keras...")
try:
    modele.save(BASE_DIR / "modele_transfer_vetements.keras")
    print("[OK] MobileNetV2 sauvegarde (.keras)")
except Exception as e:
    print(f"[WARN] Echec save .keras : {e}")
    modele.save_weights(BASE_DIR / "modele_transfer_vetements.weights.h5")
    print("[OK] Fallback : poids sauvegardes")

print("\n=== TERMINE ===")
