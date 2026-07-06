# 🖊️ OCR Manuscrit — AI Pipeline (Gemini + Mistral)

Outil CLI Python et Dashboard Streamlit pour extraire du texte depuis des images manuscrites, avec **triple moteur OCR**, **analyse IA**, et **intégration d'automatisation n8n**.

## 🏗️ Architecture

```
📷 Image → 🔧 Prétraitement → 🔍 EasyOCR (Local)
                              → ♊ Gemini Vision (API)
                              → 🌀 Mistral OCR (API)
                              → 🔗 n8n Webhook (Automation)
                                 ↓
                            🤖 AI Analysis (Gemini)
                                 ↓
           📈 Dashboard (Competition, Trust Score, Feedback)
```

## 🚀 Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Usage

### Dashboard Streamlit (Recommandé)
```bash
streamlit run app.py
```
*(Le dashboard fournit l'interface complète pour utiliser l'IA, comparer les moteurs, et envoyer vers n8n)*

### Mode CLI
```bash
# OCR basique avec tous les moteurs configurés
python ocr.py photo.jpg

# Avec détails par mot (EasyOCR)
python ocr.py photo.jpg --verbose

# Avec calcul d'erreur (CER/WER)
python ocr.py photo.jpg --ground-truth "Le texte attendu"

# Désactiver le prétraitement
python ocr.py photo.jpg --no-preprocess
```

## 🔑 Configuration des Clés API

L'application utilise les variables d'environnement suivantes. Vous pouvez également les saisir directement dans le sidebar du Dashboard Streamlit.

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "votre_cle_gemini"
$env:MISTRAL_API_KEY = "votre_cle_mistral"
$env:HF_API_TOKEN = "votre_token_hugging_face"  # Optionnel pour TrOCR
```

## 🔗 Intégration n8n

Un template de workflow `n8n_ocr_workflow.json` est fourni. 
1. Importez ce fichier dans votre instance n8n.
2. Activez le webhook test.
3. Copiez l'URL du webhook.
4. Collez l'URL dans le dashboard Streamlit sous la section "Automation".

## 📊 Métriques & Intelligence

- **Trust Score (AI)**: Note de fiabilité de 0 à 100% attribuée par Gemini sur la base d'une analyse croisée.
- **Concordance inter-moteurs**: Matrice de similarité entre tous les moteurs OCR actifs.
- **Validation NLP**: Taux de mots reconnus dans le dictionnaire, mots manquants.
- **AI Feedback**: Le texte le plus probable selon l'IA et les corrections suggérées.
