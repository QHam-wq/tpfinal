"""
Marketplace AI Moderation — Interface Streamlit
Projet Final DL TensorFlow

L'utilisateur publie une annonce (photo + titre + description + prix).
Deux modèles IA contrôlent l'annonce :
  - NLP : la description est-elle frauduleuse / spam ?
  - Vision : la photo est-elle un produit valide ?

Verdict combiné : annonce publiée ou refusée avec motifs.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

TEXT_MODEL_PATH = BASE_DIR / "modele_maison_texte.keras"
TEXT_WEIGHTS_PATH = BASE_DIR / "modele_maison_poids.weights.h5"
TEXT_VOCAB_PATH = BASE_DIR / "vocab_texte.txt"
TEXT_HF_DIR = BASE_DIR / "hf_spam_model"  # BERT-tiny finetuné spam (99%)

IMG_MAISON_PATH = BASE_DIR / "modele_maison_vetements.keras"
IMG_TRANSFER_PATH = BASE_DIR / "modele_transfer_vetements.keras"

CATEGORIES_VETEMENTS = ["Dress", "Pants", "Shirt", "Shoes", "T-Shirt"]
LABEL_FR = {
    "Dress": "Robe",
    "Pants": "Pantalon",
    "Shirt": "Chemise",
    "Shoes": "Chaussures",
    "T-Shirt": "T-Shirt",
}
NUM_CLASSES = len(CATEGORIES_VETEMENTS)
SEUIL_CONFIANCE_PHOTO = 0.60  # en-dessous : photo non conforme
SEUIL_FRAUDE_TEXTE = 0.50      # au-dessus : description suspecte

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100

st.set_page_config(
    page_title="Marketplace AI Moderation",
    page_icon="🛒",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Chargement modèle TEXTE (avec fallback poids + reconstruction)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle NLP...")
def charger_modele_texte():
    try:
        modele = tf.keras.models.load_model(TEXT_MODEL_PATH)
        _ = modele(tf.constant(["test"]))
        return modele, "complet"
    except Exception:
        pass

    if not TEXT_WEIGHTS_PATH.exists() or not TEXT_VOCAB_PATH.exists():
        raise FileNotFoundError(
            "Modèle texte indisponible. Lancez `python preparer_vocab.py` "
            "puis vérifiez la présence de modele_maison_poids.weights.h5"
        )

    with open(TEXT_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [ligne.rstrip("\n") for ligne in f]

    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        vocabulary=vocab,
    )
    modele = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        encoder,
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    modele.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    _ = modele(tf.constant(["init"]))
    modele.load_weights(TEXT_WEIGHTS_PATH)
    return modele, "reconstruit"


# -----------------------------------------------------------------------------
# Chargement modèles IMAGE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Chargement modèle TEXTE PRÉ-ENTRAÎNÉ (HuggingFace BERT-tiny finetuné SMS spam)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle texte pré-entraîné (BERT-tiny)...")
def charger_modele_texte_transfer():
    if not (TEXT_HF_DIR / "config.json").exists():
        return None
    # Imports lazy : torch est lourd, on ne le charge qu'à la demande
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(str(TEXT_HF_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(TEXT_HF_DIR))
    model.eval()
    return {"tokenizer": tok, "model": model, "torch": torch}


@st.cache_resource(show_spinner="Chargement du CNN maison...")
def _charger_modele_image_maison_cache(mtime: float):
    return tf.keras.models.load_model(IMG_MAISON_PATH)


def charger_modele_image_maison():
    if not IMG_MAISON_PATH.exists():
        return None
    return _charger_modele_image_maison_cache(IMG_MAISON_PATH.stat().st_mtime)


@st.cache_resource(show_spinner="Chargement MobileNetV2...")
def _charger_modele_image_transfer_cache(mtime: float):
    return tf.keras.models.load_model(IMG_TRANSFER_PATH)


def charger_modele_image_transfer():
    if not IMG_TRANSFER_PATH.exists():
        return None
    return _charger_modele_image_transfer_cache(IMG_TRANSFER_PATH.stat().st_mtime)


def preparer_image(image_pil: Image.Image) -> np.ndarray:
    image_pil = image_pil.convert("RGB").resize((224, 224))
    arr = np.array(image_pil, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def analyser_photo(image_pil: Image.Image):
    """
    Retourne (verdict_ok, message, details) en utilisant en priorité MobileNetV2,
    sinon le CNN maison. Verdict : True si la photo semble être un produit valide.
    """
    modele_t = charger_modele_image_transfer()
    modele_m = charger_modele_image_maison()
    modele = modele_t or modele_m
    nom_modele = "MobileNetV2 (transfer)" if modele_t else "CNN maison"

    if modele is None:
        return None, None, {"erreur": "Aucun modèle image disponible (entraînement en cours ?)"}

    x = preparer_image(image_pil)
    probas = modele.predict(x, verbose=0)[0]
    idx = int(np.argmax(probas))
    confiance = float(probas[idx])
    categorie = CATEGORIES_VETEMENTS[idx]

    details = {
        "modele": nom_modele,
        "categorie_detectee": LABEL_FR.get(categorie, categorie),
        "confiance": confiance,
        "toutes_probas": {
            LABEL_FR.get(c, c): float(probas[i]) for i, c in enumerate(CATEGORIES_VETEMENTS)
        },
    }

    if confiance >= SEUIL_CONFIANCE_PHOTO:
        return True, f"Produit identifié : **{details['categorie_detectee']}**", details
    return False, (
        "La photo ne ressemble pas à un produit reconnu "
        f"(confiance max {confiance*100:.0f} % < seuil {SEUIL_CONFIANCE_PHOTO*100:.0f} %)."
    ), details


def analyser_texte(titre: str, description: str):
    """
    Retourne (verdict_ok, message, details). True = description légitime.
    Utilise en priorité le modèle pré-entraîné HF (99%), fallback Bi-LSTM (98%).
    """
    texte = f"{titre}. {description}".strip()

    # --- 1) Modèle pré-entraîné HuggingFace BERT-tiny (prioritaire) ---
    bundle = charger_modele_texte_transfer()
    if bundle is not None:
        torch = bundle["torch"]
        tok = bundle["tokenizer"]
        model = bundle["model"]
        with torch.no_grad():
            inputs = tok(texte, return_tensors="pt", truncation=True, max_length=128)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]
        proba = float(probs[1])  # LABEL_1 = spam
        nom_modele = "BERT-tiny (pré-entraîné HF)"
    else:
        # --- 2) Fallback : Bi-LSTM maison ---
        modele, _ = charger_modele_texte()
        proba = float(modele(tf.constant([texte]))[0][0])
        nom_modele = "Bi-LSTM maison"

    details = {
        "modele": nom_modele,
        "score_fraude": proba,
        "seuil": SEUIL_FRAUDE_TEXTE,
    }

    if proba >= SEUIL_FRAUDE_TEXTE:
        return False, (
            f"Description suspecte / spam (score {proba*100:.0f} % ≥ "
            f"{SEUIL_FRAUDE_TEXTE*100:.0f} %)."
        ), details
    return True, "Description naturelle, aucun signe d'arnaque.", details


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🛒 Marketplace — Publication d'annonce")
st.caption("Modération automatique IA : texte (NLP) + photo (Computer Vision)")

# Onglets : Publier / À propos
tab_pub, tab_apropos = st.tabs(["📝 Publier une annonce", "ℹ️ À propos / Tech"])

# =============================================================================
# ONGLET PUBLIER
# =============================================================================
with tab_pub:
    col_form, col_resultat = st.columns([1, 1])

    with col_form:
        st.subheader("Détails de l'annonce")
        titre = st.text_input("Titre de l'annonce", placeholder="Ex : T-shirt Nike taille M")
        description = st.text_area(
            "Description",
            height=140,
            placeholder="Décrivez votre article : état, taille, raison de la vente...",
        )
        col_prix, col_cat = st.columns(2)
        prix = col_prix.number_input("Prix (€)", min_value=0.0, value=15.0, step=1.0)
        categorie_user = col_cat.selectbox(
            "Catégorie déclarée",
            ["Vêtements", "Chaussures", "Accessoires", "Autre"],
        )
        photo = st.file_uploader(
            "Photo de l'article", type=["jpg", "jpeg", "png", "webp"]
        )

        if photo is not None:
            st.image(photo, caption="Aperçu", width=250)

        publier = st.button(
            "🚀 Publier l'annonce", type="primary", use_container_width=True
        )

    with col_resultat:
        st.subheader("Vérification IA")

        if not publier:
            st.info(
                "Remplissez le formulaire et cliquez sur **Publier l'annonce** "
                "pour lancer la modération automatique."
            )
        else:
            # Validations basiques
            erreurs_basiques = []
            if not titre.strip():
                erreurs_basiques.append("Titre manquant.")
            if not description.strip() or len(description.strip()) < 10:
                erreurs_basiques.append(
                    "Description trop courte (10 caractères minimum)."
                )
            if photo is None:
                erreurs_basiques.append("Photo manquante.")

            if erreurs_basiques:
                st.error("⛔ Annonce incomplète :\n" + "\n".join(f"- {e}" for e in erreurs_basiques))
            else:
                with st.spinner("Analyse en cours..."):
                    # NLP
                    try:
                        ok_texte, msg_texte, det_texte = analyser_texte(titre, description)
                    except Exception as e:
                        ok_texte, msg_texte, det_texte = None, f"Erreur NLP : {e}", {}

                    # Vision
                    try:
                        image_pil = Image.open(photo)
                        ok_photo, msg_photo, det_photo = analyser_photo(image_pil)
                    except Exception as e:
                        ok_photo, msg_photo, det_photo = None, f"Erreur vision : {e}", {}

                # Affichage par module
                st.markdown("**📝 Module NLP — Description**")
                if ok_texte is True:
                    st.success(f"✅ {msg_texte}")
                elif ok_texte is False:
                    st.error(f"🚨 {msg_texte}")
                else:
                    st.warning(msg_texte)

                st.markdown("**📷 Module Vision — Photo**")
                if ok_photo is True:
                    st.success(f"✅ {msg_photo}")
                elif ok_photo is False:
                    st.error(f"🚨 {msg_photo}")
                else:
                    st.warning(
                        msg_photo or
                        "Modèle image indisponible (entraînement en cours ?)."
                    )

                st.markdown("---")

                # Verdict final
                if ok_texte is True and ok_photo is True:
                    st.success("### 🎉 Annonce publiée avec succès !")
                    st.balloons()
                    st.json({
                        "titre": titre,
                        "description": description,
                        "prix_euros": prix,
                        "categorie_declaree": categorie_user,
                        "categorie_detectee": det_photo.get("categorie_detectee"),
                    })
                elif ok_texte is False or ok_photo is False:
                    motifs = []
                    if ok_texte is False:
                        motifs.append("description signalée comme **frauduleuse / spam**")
                    if ok_photo is False:
                        motifs.append("photo **non conforme** (produit non reconnu)")
                    st.error(
                        "### 🚫 Annonce refusée\n\n"
                        "Votre annonce ne peut pas être publiée car : "
                        + " et ".join(motifs) + "."
                    )
                    st.caption(
                        "Vous pouvez corriger l'annonce (texte / photo) et réessayer."
                    )
                else:
                    st.warning(
                        "⚠️ Vérification partielle (un des modèles est indisponible). "
                        "L'annonce ne peut pas être validée automatiquement."
                    )

                # Détails techniques
                with st.expander("🔬 Détails techniques de l'analyse"):
                    st.write("**Module NLP**")
                    st.json(det_texte)
                    st.write("**Module Vision**")
                    st.json(det_photo)


# =============================================================================
# ONGLET À PROPOS
# =============================================================================
with tab_apropos:
    st.subheader("🎯 Problématique métier")
    st.markdown(
        """
La plateforme reçoit des milliers d'annonces par jour. Les modérateurs humains
sont débordés par les annonces frauduleuses (spam dans la description) et les
photos non conformes (objets illisibles, déchets, articles interdits).

**Solution IA** : un double filtre automatisé.
- 🧠 **NLP** analyse la description et bloque les arnaques.
- 👁️ **Computer Vision** vérifie la photo et n'accepte que les produits valides.

**ROI** : réduction du temps de validation de ~80 % et meilleure expérience
utilisateur.
        """
    )

    st.subheader("🛠️ Stack technique")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
**Module NLP — Détection de fraude texte**
- Dataset : SMS Spam Collection (UCI, 5572 messages)
- **Modèle maison** : `TextVectorization → Embedding(64) → Bi-LSTM(32) → Dense → Sigmoid`
  → Outils TF : `tf.data.Dataset`, `Bidirectional(LSTM)`, `Embedding(mask_zero=True)`
  → Score val : **~98 %**
- **Modèle pré-entraîné** : BERT-tiny (`mrm8488/bert-tiny-finetuned-sms-spam-detection`)
  → HuggingFace Transformers + PyTorch, fine-tuné sur SMS spam
  → Score val : **~99 %**
            """
        )
    with col2:
        st.markdown(
            f"""
**Module Vision — Validation de photo produit**
- Dataset : Fashion Marketplace (~2900 images, 5 catégories)
- Catégories : {", ".join(LABEL_FR.values())}
- Modèle maison : CNN Conv2D × 2 + Dense
- Modèle pré-entraîné : **MobileNetV2** (ImageNet) + tête custom
- Outils TF : `ImageDataGenerator`, `tf.keras.applications`, transfer learning,
  `GlobalAveragePooling2D`, `EarlyStopping`
- Seuil de confiance pour acceptation : **{int(SEUIL_CONFIANCE_PHOTO*100)} %**
            """
        )

    st.subheader("📊 Étude comparative — résultats réels")
    st.markdown(
        """
| Modèle                  | Type        | Accuracy val. | Taille fichier | Commentaire                                 |
|-------------------------|-------------|---------------|----------------|---------------------------------------------|
| CNN fait maison         | Vision      | **~61 %**     | 274 Mo         | Apprend depuis zéro, overfit dès l'epoch 3  |
| MobileNetV2 (transfer)  | Vision      | **~93 %**     | 9.3 Mo         | Pré-entraîné ImageNet → +32 pts, 30× plus léger |
| Bi-LSTM maison          | NLP texte   | **~98 %**     | 7.7 Mo (poids) | Architecture custom adaptée au SMS spam     |
| BERT-tiny (pré-entraîné)| NLP texte   | **~99 %**     | 17 Mo          | HuggingFace, fine-tuné sur SMS spam directement |

> **Lectures clés pour le rapport** :
> - Le **transfer learning** fait gagner **+32 points** d'accuracy validation
>   avec un modèle **30× plus léger** : c'est l'argument central.
> - Le CNN maison overfit rapidement (train 97 % vs val 61 %) : signe que
>   2900 images sont insuffisantes pour apprendre des features visuelles
>   robustes à partir de zéro.
> - MobileNetV2 converge dès la première epoch (val 88 %) grâce aux features
>   ImageNet déjà apprises sur des millions d'images.
        """
    )

    statut_modeles = {
        "Modèle texte maison (.keras)": TEXT_MODEL_PATH.exists(),
        "Poids texte maison (.h5)": TEXT_WEIGHTS_PATH.exists(),
        "Vocabulaire texte maison": TEXT_VOCAB_PATH.exists(),
        "Modèle texte pré-entraîné (HF BERT-tiny)": (TEXT_HF_DIR / "config.json").exists(),
        "Modèle image maison": IMG_MAISON_PATH.exists(),
        "Modèle image MobileNetV2": IMG_TRANSFER_PATH.exists(),
    }
    st.subheader("📦 État des modèles locaux")
    for nom, present in statut_modeles.items():
        st.write(f"- {'✅' if present else '⏳'} {nom}")
