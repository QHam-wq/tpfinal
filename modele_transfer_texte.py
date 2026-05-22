"""
Modele NLP PRE-ENTRAINE pour la detection de spam/fraude.

Modele : mrm8488/bert-tiny-finetuned-sms-spam-detection (HuggingFace)
         - Architecture : BERT tiny (2 couches Transformer, ~17 Mo)
         - Pre-entraine sur BookCorpus + Wikipedia
         - Fine-tune sur SMS Spam Collection par la communaute HF

Equivalent texte du transfer learning fait pour les images (MobileNetV2).

Fonctionnalites :
  - Telecharge le modele depuis HuggingFace (avec contournement SSL Windows)
  - L'evalue sur la meme validation set que le modele maison Bi-LSTM
  - Compare directement les performances
  - Fournit une fonction predire() reutilisable par l'app Streamlit

Usage :
  python modele_transfer_texte.py        # telechargement + evaluation + tests
"""

import os
import sys
import ssl
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------
# CONTOURNEMENT SSL WINDOWS
# Le poste a une chaine de CA cert cassee (Basic Constraints non critical).
# On patche httpx (utilise par huggingface_hub) ET la lib ssl globale.
# ----------------------------------------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings()

import httpx
_orig_client_init = httpx.Client.__init__
def _patched_client(self, *args, **kwargs):
    kwargs["verify"] = False
    return _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_client

_orig_async_init = httpx.AsyncClient.__init__
def _patched_async(self, *args, **kwargs):
    kwargs["verify"] = False
    return _orig_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _patched_async

# ----------------------------------------------------------------------------
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
MODELE_DIR = BASE_DIR / "hf_spam_model"
REPO_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"


# ----------------------------------------------------------------------------
# 1) Telechargement du modele (une seule fois)
# ----------------------------------------------------------------------------
def telecharger_modele():
    """Telecharge le modele BERT-tiny depuis HuggingFace s'il n'est pas deja la."""
    if (MODELE_DIR / "config.json").exists():
        print(f"[OK] Modele deja present dans {MODELE_DIR.name}/")
        return MODELE_DIR

    from huggingface_hub import snapshot_download
    print(f"[1/3] Telechargement depuis HuggingFace : {REPO_ID}")
    snapshot_download(repo_id=REPO_ID, local_dir=str(MODELE_DIR))
    print(f"[OK] Telecharge dans {MODELE_DIR}")
    return MODELE_DIR


# ----------------------------------------------------------------------------
# 2) Chargement du modele (tokenizer + classifier)
# ----------------------------------------------------------------------------
_cache = {"tokenizer": None, "model": None, "torch": None}


def charger_modele():
    """Charge le modele en memoire. Idempotent : retourne le cache si deja charge."""
    if _cache["model"] is not None:
        return _cache

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(str(MODELE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODELE_DIR))
    model.eval()  # mode inference : desactive dropout

    _cache.update({"tokenizer": tok, "model": model, "torch": torch})
    print(f"[OK] Modele charge | id2label = {model.config.id2label}")
    return _cache


# ----------------------------------------------------------------------------
# 3) Inference : predit la probabilite de spam pour un message
# ----------------------------------------------------------------------------
def predire(texte: str) -> float:
    """
    Retourne la probabilite que `texte` soit du spam, dans [0, 1].
    LABEL_1 = spam, LABEL_0 = ham.
    """
    bundle = charger_modele()
    torch = bundle["torch"]
    tok = bundle["tokenizer"]
    model = bundle["model"]

    with torch.no_grad():
        inputs = tok(texte, return_tensors="pt", truncation=True, max_length=128)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    return float(probs[1])  # proba spam


# ----------------------------------------------------------------------------
# 4) Evaluation sur le validation set (meme split que le modele maison)
# ----------------------------------------------------------------------------
def evaluer_sur_val_set():
    """Mesure accuracy / precision / recall sur le val set SMS Spam."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, confusion_matrix
    )

    print("\n[2/3] Evaluation sur le val set (meme split seed=42)...")
    df = pd.read_csv(BASE_DIR / "spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "texte"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    _, X_val, _, y_val = train_test_split(
        df["texte"], df["label"], test_size=0.2, random_state=42
    )

    preds = [1 if predire(t) >= 0.5 else 0 for t in X_val.tolist()]

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    cm = confusion_matrix(y_val, preds)

    print(f"      Accuracy  : {acc*100:.2f} %")
    print(f"      Precision : {prec*100:.2f} %  (% de vrais spam parmi les detections)")
    print(f"      Recall    : {rec*100:.2f} %  (% de spam reellement attrapes)")
    print(f"      Matrice de confusion :")
    print(f"         {'':>10} {'pred=ham':>10} {'pred=spam':>11}")
    print(f"         {'vrai=ham':>10} {cm[0,0]:>10} {cm[0,1]:>11}")
    print(f"         {'vrai=spam':>10} {cm[1,0]:>10} {cm[1,1]:>11}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "confusion": cm}


# ----------------------------------------------------------------------------
# 5) Demo : tests rapides sur quelques messages
# ----------------------------------------------------------------------------
def tester_exemples():
    print("\n[3/3] Tests sur des exemples concrets :")
    exemples = [
        "WINNER!! You have won a 1000 dollars prize. Call now!",
        "Hey are we still meeting at 6pm for dinner?",
        "Free entry in 2 a wkly comp to win FA Cup tickets",
        "Your account has been suspended. Click http://bit.ly/x",
        "Vends mon t-shirt Nike taille M, etat impeccable",
        "URGENT: Click here NOW to verify your account or it will be suspended",
    ]
    for t in exemples:
        s = predire(t)
        verdict = "SPAM" if s >= 0.5 else "HAM "
        print(f"  [{verdict}] score={s:.3f}  | {t[:60]}")


# ----------------------------------------------------------------------------
# Point d'entree
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    telecharger_modele()
    charger_modele()
    evaluer_sur_val_set()
    tester_exemples()
    print("\n=== TERMINE ===")
