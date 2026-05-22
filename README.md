# 🛒 Marketplace AI Moderation

> Projet Final — Deep Learning sous TensorFlow
> Double filtre IA pour la modération automatique d'annonces de type Vinted / Leboncoin / Facebook Marketplace.

---

## 🎯 Problématique métier

Les plateformes de revente reçoivent **des milliers d'annonces par jour**. Les modérateurs humains sont débordés par :
- 🚨 les **descriptions frauduleuses** (spam, arnaques, phishing)
- 📸 les **photos non conformes** (objets illisibles, articles interdits, photos hors sujet)

### Solution proposée

Un **double filtre automatisé** basé sur deux modules d'IA :
- 🧠 **Module NLP** : analyse la description pour bloquer les arnaques
- 👁️ **Module Vision** : vérifie la photo et n'accepte que les produits valides

L'annonce n'est publiée que si **les deux modules valident**.

### ROI estimé
- ⏱️ Réduction du temps de validation manuelle de **~80 %**
- 😊 Meilleure expérience utilisateur (réponse instantanée)
- 🛡️ Protection contre les arnaques avant publication

---

## 📂 Architecture du projet

```
projet final TF/
│
├── 📱 app.py                          # Interface Streamlit "marketplace" (entry point)
│
├── 📝 MODULE TEXTE (NLP)
│   ├── preparer_vocab.py              # Génère le vocabulaire pour le Bi-LSTM maison
│   ├── projet_final.ipynb             # Notebook initial — entraînement Bi-LSTM
│   ├── modele_transfer_texte.py       # Modèle pré-entraîné HF (BERT-tiny)
│   │
│   ├── vocab_texte.txt                # 8 452 tokens du Bi-LSTM
│   ├── modele_maison_texte.keras      # Modèle Bi-LSTM (fichier corrompu, voir .h5)
│   ├── modele_maison_poids.weights.h5 # Poids du Bi-LSTM (utilisé en production)
│   ├── hf_spam_model/                 # BERT-tiny téléchargé depuis HuggingFace
│   └── spam.csv                       # Dataset SMS Spam Collection (5572 messages)
│
├── 👁️ MODULE VISION
│   ├── entrainer_vetements.py         # Entraîne le CNN fait maison
│   ├── entrainer_mobilenet.py         # Entraîne MobileNetV2 (transfer learning)
│   │
│   ├── modele_maison_vetements.keras  # CNN fait maison (274 Mo)
│   ├── modele_transfer_vetements.keras # MobileNetV2 fine-tuné (9.3 Mo)
│   ├── images_original/               # 5 762 images d'articles de mode
│   └── images.csv                     # Labels des images (image, label)
│
├── 📦 ENVIRONNEMENT
│   ├── .venv/                         # Environnement Python virtuel
│   └── README.md                      # Vous êtes ici
```

---

## 🧩 Description fichier par fichier

### Interface

| Fichier | Rôle |
|---|---|
| **`app.py`** | Application Streamlit principale. Formulaire d'annonce (titre, description, prix, catégorie, photo). Verdict combiné : ✅ publiée / 🚫 refusée. Onglet "À propos" avec stack technique et étude comparative. |

### Module Texte (NLP)

| Fichier | Rôle |
|---|---|
| **`projet_final.ipynb`** | Notebook initial : exploration du dataset SMS Spam, construction et entraînement du **Bi-LSTM fait maison** (TextVectorization → Embedding(64) → Bi-LSTM(32) → Dense → Sigmoid). |
| **`preparer_vocab.py`** | Génère `vocab_texte.txt` à partir du dataset SMS Spam. Permet à l'app de reconstruire le `TextVectorization` sans dépendre de Kaggle au runtime. |
| **`modele_transfer_texte.py`** | Télécharge depuis HuggingFace le modèle **`mrm8488/bert-tiny-finetuned-sms-spam-detection`** (BERT-tiny déjà fine-tuné sur SMS spam). Inclut : téléchargement avec contournement SSL Windows, évaluation sur le val set, fonction `predire()` réutilisable. |
| **`vocab_texte.txt`** | Vocabulaire de 8 452 tokens appris par le `TextVectorization` du Bi-LSTM maison. |
| **`modele_maison_poids.weights.h5`** | Poids entraînés du Bi-LSTM maison (~7.7 Mo). Chargés via reconstruction d'architecture dans l'app. |
| **`hf_spam_model/`** | Dossier du modèle BERT-tiny téléchargé : `config.json`, `model.safetensors`, `tokenizer_config.json`, `vocab.txt`. |
| **`spam.csv`** | Dataset SMS Spam Collection (UCI) — 5 572 messages, ratio 87 % ham / 13 % spam. |

### Module Vision

| Fichier | Rôle |
|---|---|
| **`entrainer_vetements.py`** | Entraîne le **CNN fait maison** sur le dataset Fashion (Rescaling + 2× Conv2D/MaxPool + Flatten + Dense). EarlyStopping + sauvegarde `.keras`. |
| **`entrainer_mobilenet.py`** | Entraîne le **MobileNetV2 pré-entraîné** (poids ImageNet figés) avec une tête custom (GlobalAveragePooling + Dropout + Dense). Inclut contournement SSL pour le téléchargement des poids ImageNet. |
| **`modele_maison_vetements.keras`** | CNN fait maison (274 Mo, lourd à cause du Flatten + Dense(128)). |
| **`modele_transfer_vetements.keras`** | MobileNetV2 fine-tuné (9.3 Mo, ultra-compact grâce au GlobalAveragePooling). |
| **`images_original/`** | Dataset Fashion Marketplace, 5 762 photos d'articles classées en 5 catégories : `T-Shirt`, `Shoes`, `Pants`, `Shirt`, `Dress`. |
| **`images.csv`** | Mapping `image_id → label` pour les 5 762 photos. |

---

## 🛠️ Stack technique

### Framework principal
- **TensorFlow / Keras 2.21** — modèles maison (Bi-LSTM, CNN, MobileNetV2)
- **PyTorch 2.12** — backend pour le modèle BERT-tiny HuggingFace
- **HuggingFace Transformers 5.9** — chargement du modèle pré-entraîné texte
- **Streamlit 1.57** — interface utilisateur

### Outils TensorFlow utilisés (point 5 de l'énoncé)
| Outil TF | Où | Pourquoi |
|---|---|---|
| `tf.data.Dataset` | NLP train pipeline | Pipeline performant avec batch/cache/prefetch |
| `TextVectorization` | NLP maison | Tokenization intégrée au graphe |
| `Embedding(mask_zero=True)` | NLP maison | Embeddings denses + masque de padding |
| `Bidirectional(LSTM)` | NLP maison | Lecture séquentielle bidirectionnelle |
| `ImageDataGenerator.flow_from_dataframe` | Vision | Pipeline images depuis CSV |
| `tf.keras.applications.MobileNetV2` | Vision transfer | Backbone ImageNet pré-entraîné |
| `applications.mobilenet_v2.preprocess_input` | Vision transfer | Normalisation adaptée au modèle |
| `GlobalAveragePooling2D` | Vision transfer | Tête de classification compacte |
| `callbacks.EarlyStopping(restore_best_weights=True)` | Tous | Régularisation par arrêt anticipé |

---

## 📊 Étude comparative — Modèles fait maison vs pré-entraînés

### Module Vision

| Critère | 🏠 CNN Fait Maison | 🚀 MobileNetV2 Pré-entraîné |
|---|---|---|
| **Architecture** | Conv2D × 2 + MaxPool + Flatten + Dense | MobileNetV2 (ImageNet) + GlobalAvgPool + Dense |
| **Paramètres** | ~24 M (Flatten lourd) | ~2.3 M backbone + 6 k tête |
| **Taille fichier** | 274 Mo | **9.3 Mo** (30× plus léger) |
| **Train accuracy** | 99.5 % | 97.3 % |
| **Val accuracy** | 60.9 % | **93.0 %** (+32 pts) |
| **Diagnostic** | Surapprentissage massif | Apprentissage stable |
| **Verdict** | Baseline (prouve qu'un réseau simple ne suffit pas) | **Modèle retenu en production** |

### Module Texte (NLP)

| Critère | 🏠 Bi-LSTM Fait Maison | 🚀 BERT-tiny Pré-entraîné |
|---|---|---|
| **Architecture** | TextVectorization → Embedding(64) → Bi-LSTM(32) → Dense | BERT-tiny (2 Transformer blocks) |
| **Paramètres** | 666 945 | ~4.4 M |
| **Taille fichier** | 7.7 Mo (poids) | 17 Mo (.safetensors) |
| **Val accuracy** | 98.21 % | **99.01 %** (+0.8 pts) |
| **Val precision (spam)** | 95.14 % | **97.93 %** |
| **Val recall (spam)** | 91.33 % | **94.67 %** |
| **Faux négatifs** | 13/150 | **8/150** |
| **Faux positifs** | 7/965 | **3/965** |
| **Verdict** | Baseline custom, code transparent | **Modèle retenu en production** (Bi-LSTM en fallback) |

### Lecture clé pour le rapport

> Le transfer learning bat systématiquement le "fait maison" sur ce projet :
> - **+32 pts** en vision (le gap est massif car le dataset est petit pour apprendre des features visuelles)
> - **+0.8 pts** en NLP (gap modeste car le SMS spam est une tâche relativement simple où un Bi-LSTM bien conçu suffit)
> Cette différence d'amplitude est elle-même une **conclusion intéressante** : le transfer learning est d'autant plus utile que la tâche est complexe et le dataset petit.

---

## 🚀 Installation & Lancement

### 1. Prérequis
- Python 3.13+
- Windows / macOS / Linux (testé sur Windows 11)
- ~3 Go d'espace disque (dataset images + modèles + dépendances)

### 2. Installation des dépendances

```bash
# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate     # macOS/Linux

# Installer les paquets
pip install tensorflow streamlit pillow scikit-learn pandas
pip install transformers huggingface-hub
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "setuptools<81"    # nécessaire pour tensorflow-hub
```

### 3. Préparer les artefacts (à faire une seule fois)

```bash
# Vocabulaire du Bi-LSTM maison
python preparer_vocab.py

# Modèle pré-entraîné texte
python modele_transfer_texte.py

# Modèles image (entraînement ~30-60 min CPU)
python entrainer_vetements.py
python entrainer_mobilenet.py
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

→ Ouvre http://localhost:8501

---

## 📁 Datasets utilisés

| Dataset | Source | Volume | Usage |
|---|---|---|---|
| **SMS Spam Collection** | [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (via Kaggle) | 5 572 SMS | Module NLP |
| **Fashion Marketplace** | Dataset interne d'images de mode | 5 762 images, 5 catégories | Module Vision |
| **ImageNet** (transfer learning) | Google (via `tf.keras.applications`) | 1.4 M images, 1000 classes | Backbone MobileNetV2 |
| **Google News + Wikipedia + BookCorpus** | Pré-entraînement de BERT-tiny | 3 milliards de mots | Backbone BERT-tiny |

---

## ⚠️ Notes techniques

### Contournement SSL Windows
Sur certains postes Windows, la chaîne de certificats CA est mal configurée (`Basic Constraints not marked critical`), ce qui bloque les téléchargements depuis `storage.googleapis.com` (poids MobileNetV2) et `huggingface.co` (BERT-tiny). Les scripts `entrainer_mobilenet.py` et `modele_transfer_texte.py` incluent un **contournement** via `ssl._create_unverified_context` et un patch sur `httpx.Client`.

### Modèle Keras corrompu
Le fichier `modele_maison_texte.keras` est corrompu (8 Ko au lieu des 2.5 Mo attendus). L'app utilise donc un **fallback** : reconstruction de l'architecture + chargement des poids depuis `modele_maison_poids.weights.h5`.
