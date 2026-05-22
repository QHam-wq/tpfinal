"""
Script à lancer UNE FOIS pour préparer le vocabulaire du modèle texte.

Génère 'vocab_texte.txt' à partir du dataset SMS Spam (déjà téléchargé via
kagglehub). Ce vocab sera ensuite chargé directement par l'app Streamlit,
sans avoir besoin de re-télécharger le dataset.
"""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import shutil
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
CSV_LOCAL = BASE_DIR / "spam.csv"
VOCAB_PATH = BASE_DIR / "vocab_texte.txt"

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100

# --- 1. Récupérer le CSV (cache kagglehub déjà rempli) ---
if not CSV_LOCAL.exists():
    import kagglehub
    chemin = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    shutil.copy(Path(chemin) / "spam.csv", CSV_LOCAL)
    print(f"[OK] CSV copie dans {CSV_LOCAL}")
else:
    print(f"[INFO] CSV deja present ({CSV_LOCAL})")

# --- 2. Préparer les données comme dans le notebook ---
df = pd.read_csv(CSV_LOCAL, encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "texte"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, _, _, _ = train_test_split(
    df["texte"], df["label"], test_size=0.2, random_state=42
)

# --- 3. Adapter l'encodeur ---
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_SEQUENCE_LENGTH,
)
encoder.adapt(X_train.values)

# --- 4. Sauvegarder le vocabulaire ---
vocab = encoder.get_vocabulary()
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    for mot in vocab:
        f.write(mot + "\n")

print(f"[OK] Vocabulaire sauvegarde ({len(vocab)} tokens) dans {VOCAB_PATH}")
