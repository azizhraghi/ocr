# 🖊️ OCR Manuscrit — Extracteur de texte manuscrit

Outil CLI Python pour extraire du texte depuis des images manuscrites, avec **double moteur OCR**, **validation NLP**, et **métriques de qualité**.

## 🏗️ Architecture

```
📷 Image → 🔧 Prétraitement → 🔍 EasyOCR (confiance) → 📊 NLP Validation → 📈 Dashboard
                              → 🌐 TrOCR API (optionnel) ↗
```

### Moteurs OCR
| Moteur | Type | Confiance | Prérequis |
|--------|------|-----------|-----------|
| **EasyOCR** | Local (toujours actif) | ✅ Score 0-1 par mot | Aucun |
| **TrOCR** | API Hugging Face (optionnel) | ❌ | `HF_API_TOKEN` |

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📖 Usage

```bash
# OCR basique
python ocr.py photo.jpg

# Avec détails par mot
python ocr.py photo.jpg --verbose

# Avec calcul d'erreur (CER/WER)
python ocr.py photo.jpg --ground-truth "Le texte attendu"

# Multi-langue (anglais + français)
python ocr.py photo.jpg --lang en fr

# Désactiver le prétraitement
python ocr.py photo.jpg --no-preprocess
```

## 📊 Métriques de sortie

| Métrique | Description |
|----------|-------------|
| **Confiance moyenne** | Score moyen de confiance EasyOCR (0-100%) |
| **Confiance minimale** | Pire score de confiance |
| **Taux dictionnaire** | % de mots reconnus dans le dictionnaire |
| **Grade (A-F)** | Note composite de lisibilité |
| **CER** | Character Error Rate (si `--ground-truth` fourni) |
| **WER** | Word Error Rate (si `--ground-truth` fourni) |
| **Concordance** | Similarité entre EasyOCR et TrOCR |

## 🔑 Activer TrOCR (optionnel)

```bash
# Linux/Mac
export HF_API_TOKEN="hf_votre_token_ici"

# Windows PowerShell
$env:HF_API_TOKEN = "hf_votre_token_ici"

# Windows CMD
set HF_API_TOKEN=hf_votre_token_ici
```

Obtenez un token gratuit sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## 📋 Dépendances

- `easyocr` — Moteur OCR principal
- `requests` — Appels API HF
- `Pillow` — Manipulation d'images
- `opencv-python-headless` — Prétraitement
- `jiwer` — Métriques CER/WER (optionnel, fallback intégré)
