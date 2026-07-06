#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  OCR Manuscrit — Extracteur de texte manuscrit avec validation NLP
=============================================================================

Multi-engine OCR pipeline: EasyOCR (local), Gemini Vision (API),
Mistral OCR (API), with NLP validation, quality metrics, and
AI-powered analysis.

Usage:
    python ocr.py image.jpg
    python ocr.py image.jpg --ground-truth "Hello World"
    python ocr.py image.jpg --verbose
    python ocr.py image.jpg --lang en fr

Dépendances: easyocr, requests, Pillow, opencv-python-headless, jiwer,
             google-genai, mistralai
"""

import argparse
import base64
import difflib
import io
import json
import math
import os
import re
import string
import sys
import time
from pathlib import Path

# ── Fix Windows cp1252 encoding: force UTF-8 output ──
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Imports tiers — avec messages d'erreur clairs si manquants
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("❌ Erreur: opencv-python-headless n'est pas installé.")
    print("   → pip install opencv-python-headless")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ Erreur: Pillow n'est pas installé.")
    print("   → pip install Pillow")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌ Erreur: requests n'est pas installé.")
    print("   → pip install requests")
    sys.exit(1)

# ── FIX: Monkey-patch urllib.request to use requests for EasyOCR model downloads ──
# This fixes the [Errno 11001] getaddrinfo failed error on some Windows networks
import urllib.request
def custom_urlretrieve(url, filename, reporthook=None, data=None):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename, response.headers
    except Exception as e:
        print(f"Custom download failed for {url}: {e}")
        raise

def custom_urlopen(url, data=None, timeout=None):
    try:
        resp = requests.get(url if isinstance(url, str) else url.get_full_url(), timeout=timeout or 30)
        resp.raise_for_status()
        class DummyResponse:
            def __init__(self, content): self.content = content
            def read(self): return self.content
        return DummyResponse(resp.content)
    except Exception as e:
        print(f"Custom urlopen failed: {e}")
        raise

urllib.request.urlretrieve = custom_urlretrieve
urllib.request.urlopen = custom_urlopen

try:
    import easyocr
except ImportError:
    print("❌ Erreur: easyocr n'est pas installé.")
    print("   → pip install easyocr")
    sys.exit(1)

# jiwer est optionnel — utilisé seulement si --ground-truth est fourni
try:
    from jiwer import cer, wer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

# Gemini — optionnel
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Mistral — optionnel
try:
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
TROCR_API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-large-handwritten"
MAX_API_RETRIES = 5

# Dictionnaire anglais minimal intégré (~1000 mots courants)
# Utilisé pour la validation NLP sans dépendance externe
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    "been", "being", "has", "had", "did", "does", "doing", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "am",
    "hello", "world", "name", "please", "thank", "thanks", "yes", "yeah",
    "ok", "okay", "sorry", "help", "need", "right", "left", "here",
    "where", "why", "how", "much", "many", "more", "less", "very", "too",
    "really", "great", "nice", "beautiful", "love", "happy", "sad",
    "big", "small", "long", "short", "old", "young", "high", "low",
    "open", "close", "start", "stop", "begin", "end", "read", "write",
    "call", "run", "walk", "talk", "play", "eat", "drink", "sleep",
    "live", "die", "buy", "sell", "pay", "keep", "let", "put", "tell",
    "ask", "try", "leave", "turn", "show", "hear", "seem", "feel",
    "hand", "eye", "head", "face", "body", "foot", "life", "world",
    "school", "home", "house", "room", "door", "water", "food", "book",
    "word", "letter", "number", "part", "place", "case", "point", "thing",
    "man", "woman", "child", "boy", "girl", "mother", "father", "family",
    "friend", "city", "country", "state", "company", "group", "problem",
    "fact", "money", "game", "story", "power", "change", "move", "set",
    "line", "class", "note", "test", "mark", "order", "data", "plan",
    "care", "free", "real", "sure", "best", "better", "last", "next",
    "each", "every", "both", "few", "own", "such", "same", "different",
    "kind", "still", "again", "never", "always", "often", "ever",
    "dear", "doctor", "patient", "date", "address", "phone", "email",
    "meeting", "today", "tomorrow", "yesterday", "morning", "evening",
    "night", "week", "month", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november",
    "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "mr", "mrs", "miss", "sir", "madam",
    "important", "urgent", "private", "public", "special", "general",
    "possible", "available", "necessary", "clear", "certain", "true",
    "false", "full", "empty", "easy", "hard", "simple", "difficult",
    # Mots français courants (bonus)
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "est",
    "en", "que", "qui", "dans", "ce", "il", "pas", "ne", "plus",
    "par", "sur", "au", "avec", "se", "son", "sa", "ses", "tout",
    "mais", "ou", "comme", "on", "fait", "bien", "dire", "elle",
    "avant", "deux", "aussi", "bon", "jour", "bonjour", "merci",
    "oui", "non", "cher", "madame", "monsieur",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRÉTRAITEMENT D'IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> "numpy.ndarray":
    """
    Charge et prétraite une image pour améliorer la reconnaissance OCR.
    
    Pipeline:
    1. Chargement de l'image
    2. Conversion en niveaux de gris
    3. Amélioration du contraste (CLAHE)
    4. Débruitage léger
    
    Returns:
        Image prétraitée (numpy array)
    """
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image: {image_path}")
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Améliorer le contraste avec CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Très efficace pour le texte manuscrit sur fond inégal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Débruitage léger — préserve les bords (important pour le texte)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
    
    return denoised


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOTEUR OCR — EasyOCR (local, avec confiance)
# ─────────────────────────────────────────────────────────────────────────────
def ocr_easyocr(image_path: str, languages: list[str] = None) -> dict:
    """
    Exécute EasyOCR sur une image.
    
    Returns:
        dict avec:
        - text: texte complet extrait
        - words: liste de dicts {text, confidence, bbox}
        - avg_confidence: confiance moyenne
        - min_confidence: confiance minimale
        - engine: "easyocr"
        - elapsed_seconds: temps d'exécution
    """
    if languages is None:
        languages = ["en"]
    
    print("🔍 EasyOCR — Analyse en cours...")
    start_time = time.time()
    
    # Initialiser le reader (le modèle est mis en cache après le 1er appel)
    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    
    # Exécuter l'OCR — detail=1 donne (bbox, text, confidence)
    results = reader.readtext(image_path)
    elapsed = time.time() - start_time
    
    if not results:
        return {
            "text": "",
            "words": [],
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "engine": "easyocr",
            "elapsed_seconds": elapsed,
        }
    
    # Extraire les résultats mot par mot
    words = []
    for bbox, text, confidence in results:
        words.append({
            "text": text,
            "confidence": round(confidence, 4),
            "bbox": bbox,
        })
    
    # Construire le texte complet
    full_text = " ".join(w["text"] for w in words)
    
    # Statistiques de confiance
    confidences = [w["confidence"] for w in words]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    min_conf = min(confidences) if confidences else 0.0
    
    print(f"   ✅ Terminé en {elapsed:.1f}s — {len(words)} segment(s) détecté(s)")
    
    return {
        "text": full_text,
        "words": words,
        "avg_confidence": round(avg_conf, 4),
        "min_confidence": round(min_conf, 4),
        "engine": "easyocr",
        "elapsed_seconds": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.B. MOTEUR OCR — Gemini Vision (Google AI API)
# ─────────────────────────────────────────────────────────────────────────────
def ocr_gemini(image_path: str, api_key: str = None) -> dict:
    """
    Exécute Gemini Vision OCR sur une image manuscrite.
    
    Utilise le modèle gemini-2.0-flash avec vision pour extraire
    le texte manuscrit d'une image.
    
    Returns:
        dict avec:
        - text: texte extrait
        - engine: "gemini"
        - elapsed_seconds: temps d'exécution
        - model: nom du modèle utilisé
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ℹ️  Gemini — Ignoré (GEMINI_API_KEY non défini)")
        return None
    
    if not GEMINI_AVAILABLE:
        print("❌ Gemini — google-genai n'est pas installé")
        print("   → pip install google-genai")
        return None
    
    print("♊ Gemini Vision — Analyse en cours...")
    start_time = time.time()
    
    try:
        # Initialize the client
        client = genai.Client(api_key=api_key)
        
        # Read and encode the image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Detect MIME type
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".webp": "image/webp", ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        mime_type = mime_map.get(ext, "image/png")
        
        # Create the image part
        image_part = genai.types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )
        
        # Prompt optimized for handwritten text extraction
        prompt = (
            "You are an expert OCR system specialized in handwritten text recognition. "
            "Extract ALL text visible in this image exactly as written. "
            "Do not interpret, correct, or modify the text — transcribe it character by character. "
            "If you can't read a word clearly, provide your best guess with [?] after it. "
            "Return ONLY the extracted text, nothing else."
        )
        
        # Call the API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part],
        )
        
        extracted_text = response.text.strip() if response.text else ""
        elapsed = time.time() - start_time
        
        print(f"   ✅ Terminé en {elapsed:.1f}s")
        
        return {
            "text": extracted_text,
            "engine": "gemini",
            "model": "gemini-2.0-flash",
            "elapsed_seconds": round(elapsed, 2),
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Erreur Gemini: {e}")
        return {
            "text": f"Erreur Gemini: {e}",
            "engine": "gemini",
            "model": "gemini-2.0-flash",
            "elapsed_seconds": round(elapsed, 2),
            "error": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2.C. MOTEUR OCR — Mistral OCR (Mistral AI API)
# ─────────────────────────────────────────────────────────────────────────────
def ocr_mistral(image_path: str, api_key: str = None) -> dict:
    """
    Exécute Mistral OCR sur une image manuscrite.
    
    Utilise le modèle mistral-ocr-latest pour l'extraction de texte.
    
    Returns:
        dict avec:
        - text: texte extrait
        - engine: "mistral"
        - elapsed_seconds: temps d'exécution
        - model: nom du modèle utilisé
        - pages: nombre de pages traitées
    """
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ℹ️  Mistral — Ignoré (MISTRAL_API_KEY non défini)")
        return None
    
    if not MISTRAL_AVAILABLE:
        print("❌ Mistral — mistralai n'est pas installé")
        print("   → pip install mistralai")
        return None
    
    print("🌀 Mistral OCR — Analyse en cours...")
    start_time = time.time()
    
    try:
        # Initialize the client
        client = Mistral(api_key=api_key)
        
        # Read and encode the image as base64 data URI
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".webp": "image/webp", ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        mime_type = mime_map.get(ext, "image/png")
        
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_data}"
        
        # Call Mistral OCR API
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": data_uri,
            },
        )
        
        # Extract text from all pages
        texts = []
        if ocr_response and hasattr(ocr_response, 'pages'):
            for page in ocr_response.pages:
                if hasattr(page, 'markdown') and page.markdown:
                    texts.append(page.markdown)
                elif hasattr(page, 'text') and page.text:
                    texts.append(page.text)
        
        full_text = "\n".join(texts).strip()
        elapsed = time.time() - start_time
        
        page_count = len(ocr_response.pages) if hasattr(ocr_response, 'pages') else 0
        
        print(f"   ✅ Terminé en {elapsed:.1f}s — {page_count} page(s) traitée(s)")
        
        return {
            "text": full_text,
            "engine": "mistral",
            "model": "mistral-ocr-latest",
            "elapsed_seconds": round(elapsed, 2),
            "pages": page_count,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Erreur Mistral: {e}")
        return {
            "text": f"Erreur Mistral: {e}",
            "engine": "mistral",
            "model": "mistral-ocr-latest",
            "elapsed_seconds": round(elapsed, 2),
            "error": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2.D. MOTEUR OCR — TrOCR via Hugging Face Inference API (optionnel)
# ─────────────────────────────────────────────────────────────────────────────
def ocr_trocr_api(image_path: str) -> dict | None:
    """
    Envoie l'image au modèle TrOCR via l'API d'inférence Hugging Face.
    
    - Gère le cold start (503) avec retry + estimated_time
    - Maximum 5 tentatives
    - Retourne None si HF_API_TOKEN n'est pas défini
    
    Returns:
        dict avec:
        - text: texte extrait
        - engine: "trocr-hf-api"
        - attempts: nombre de tentatives
        ou None si indisponible
    """
    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        print("ℹ️  TrOCR — Ignoré (HF_API_TOKEN non défini)")
        print("   → Définir HF_API_TOKEN pour activer le double moteur")
        return None
    
    print("🌐 TrOCR (HF API) — Envoi en cours...")
    start_time = time.time()
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Lire l'image en bytes bruts (pas de base64, pas de JSON)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            response = requests.post(
                TROCR_API_URL,
                headers=headers,
                data=image_bytes,
                timeout=60,
            )
            
            if response.status_code == 200:
                # Parser la réponse — peut être liste ou dict
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "")
                elif isinstance(data, dict):
                    text = data.get("generated_text", "")
                else:
                    text = str(data)
                
                elapsed = time.time() - start_time
                print(f"   ✅ Terminé en {elapsed:.1f}s (tentative {attempt}/{MAX_API_RETRIES})")
                
                return {
                    "text": text.strip(),
                    "engine": "trocr-hf-api",
                    "attempts": attempt,
                    "elapsed_seconds": round(elapsed, 2),
                }
            
            elif response.status_code == 503:
                # Modèle en cold start — attendre le temps estimé
                try:
                    error_data = response.json()
                    wait_time = error_data.get("estimated_time", 20)
                except (json.JSONDecodeError, ValueError):
                    wait_time = 20
                
                print(f"   ⏳ Modèle en chargement... attente {wait_time:.0f}s "
                      f"(tentative {attempt}/{MAX_API_RETRIES})")
                time.sleep(wait_time)
            
            else:
                # Erreur inattendue
                error_msg = response.text[:200]
                print(f"   ❌ Erreur HTTP {response.status_code}: {error_msg}")
                print(f"   → TrOCR désactivé pour cette exécution")
                return None
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️ Timeout (tentative {attempt}/{MAX_API_RETRIES})")
        except requests.exceptions.ConnectionError:
            print(f"   🔌 Erreur de connexion (tentative {attempt}/{MAX_API_RETRIES})")
            time.sleep(5)
    
    print(f"   ❌ Échec après {MAX_API_RETRIES} tentatives")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. AI ANALYSIS — Analyse intelligente des résultats OCR
# ─────────────────────────────────────────────────────────────────────────────
def analyze_with_ai(
    ocr_results: dict,
    image_path: str = None,
    api_key: str = None,
) -> dict:
    """
    Utilise Gemini pour analyser les résultats OCR et fournir:
    - La meilleure interprétation du texte
    - Des corrections suggérées
    - Un score de confiance global
    - Un classement des moteurs
    
    Args:
        ocr_results: dict avec les clés engine_name -> result_dict
        image_path: chemin vers l'image originale (optionnel, pour re-analyse)
        api_key: clé API Gemini
    
    Returns:
        dict avec analysis, corrections, trust_score, engine_ranking
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key or not GEMINI_AVAILABLE:
        return {
            "best_text": "",
            "corrections": [],
            "trust_score": 0.0,
            "engine_ranking": [],
            "interpretation": "AI analysis unavailable — Gemini API key required.",
            "error": True,
        }
    
    print("🤖 AI Analysis — Analyse intelligente en cours...")
    start_time = time.time()
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Build the analysis prompt with all OCR results
        engines_text = ""
        for engine_name, result in ocr_results.items():
            if result and isinstance(result, dict) and result.get("text"):
                text = result["text"]
                conf = result.get("avg_confidence", "N/A")
                engines_text += f"\n### {engine_name}\n"
                engines_text += f"Text: \"{text}\"\n"
                if conf != "N/A":
                    engines_text += f"Average Confidence: {conf}\n"
        
        if not engines_text.strip():
            return {
                "best_text": "",
                "corrections": [],
                "trust_score": 0.0,
                "engine_ranking": [],
                "interpretation": "No OCR results to analyze.",
                "error": True,
            }
        
        prompt = f"""You are an expert handwriting analyst and OCR quality assessor.

Multiple OCR engines have extracted text from the same handwritten image. 
Analyze their outputs and provide your expert assessment.

## OCR Results:
{engines_text}

## Your Task:
Respond in this exact JSON format (no markdown fencing, just raw JSON):
{{
    "best_text": "The most accurate version of the handwritten text, combining the best parts of all engines",
    "corrections": ["List of specific corrections you made and why"],
    "trust_score": 0.85,
    "engine_ranking": [
        {{"engine": "engine_name", "score": 0.9, "reason": "Why this engine performed well/poorly"}},
    ],
    "interpretation": "What the handwritten text likely means or says, with context about quality and readability"
}}

Important:
- trust_score is 0.0 (completely unreadable) to 1.0 (perfectly clear)
- Be honest — if the text is unclear, say so
- Rank ALL engines that provided results
- corrections should be specific, e.g. "Changed 'teh' to 'the' — common OCR transposition"
"""
        
        # Optionally include the image for re-analysis
        contents = [prompt]
        if image_path and os.path.isfile(image_path):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            ext = Path(image_path).suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".bmp": "image/bmp",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(ext, "image/png")
            image_part = genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            contents = [prompt, image_part]
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
        )
        
        elapsed = time.time() - start_time
        raw_text = response.text.strip() if response.text else "{}"
        
        # Parse the JSON response
        # Strip markdown code fencing if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines)
        
        try:
            analysis = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                except json.JSONDecodeError:
                    analysis = {
                        "best_text": raw_text,
                        "corrections": [],
                        "trust_score": 0.5,
                        "engine_ranking": [],
                        "interpretation": raw_text,
                    }
            else:
                analysis = {
                    "best_text": raw_text,
                    "corrections": [],
                    "trust_score": 0.5,
                    "engine_ranking": [],
                    "interpretation": raw_text,
                }
        
        analysis["elapsed_seconds"] = round(elapsed, 2)
        analysis["error"] = False
        
        print(f"   ✅ Analyse terminée en {elapsed:.1f}s")
        return analysis
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Erreur AI Analysis: {e}")
        return {
            "best_text": "",
            "corrections": [],
            "trust_score": 0.0,
            "engine_ranking": [],
            "interpretation": f"AI analysis failed: {e}",
            "elapsed_seconds": round(elapsed, 2),
            "error": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDATION NLP
# ─────────────────────────────────────────────────────────────────────────────
def clean_word(word: str) -> str:
    """Nettoie un mot : minuscule, supprime ponctuation aux extrémités."""
    return word.lower().strip(string.punctuation + string.whitespace)


def validate_nlp(text: str) -> dict:
    """
    Valide le texte extrait avec des techniques NLP :
    
    1. Dictionary Hit Rate — % de mots trouvés dans le dictionnaire
    2. Spell Suggestions — corrections suggérées pour les mots inconnus
    3. Text Statistics — nombre de mots, caractères, lignes
    
    Returns:
        dict avec toutes les métriques NLP
    """
    if not text or not text.strip():
        return {
            "dictionary_hit_rate": 0.0,
            "known_words": [],
            "unknown_words": [],
            "suggestions": {},
            "word_count": 0,
            "char_count": 0,
            "has_numbers": False,
            "has_special_chars": False,
        }
    
    # Tokeniser le texte en mots
    raw_words = text.split()
    cleaned = [clean_word(w) for w in raw_words]
    cleaned = [w for w in cleaned if w]  # Supprimer les vides
    
    if not cleaned:
        return {
            "dictionary_hit_rate": 0.0,
            "known_words": [],
            "unknown_words": [],
            "suggestions": {},
            "word_count": 0,
            "char_count": len(text),
            "has_numbers": bool(re.search(r'\d', text)),
            "has_special_chars": bool(re.search(r'[^a-zA-Z0-9\s]', text)),
        }
    
    # Vérifier chaque mot dans le dictionnaire
    known = []
    unknown = []
    suggestions = {}
    
    for word in cleaned:
        # Les nombres sont considérés comme "connus"
        if word.isdigit():
            known.append(word)
            continue
        
        if word in COMMON_WORDS:
            known.append(word)
        else:
            unknown.append(word)
            # Trouver des suggestions via difflib (fuzzy matching)
            matches = difflib.get_close_matches(word, COMMON_WORDS, n=3, cutoff=0.6)
            if matches:
                suggestions[word] = matches
    
    hit_rate = len(known) / len(cleaned) if cleaned else 0.0
    
    return {
        "dictionary_hit_rate": round(hit_rate, 4),
        "known_words": known,
        "unknown_words": unknown,
        "suggestions": suggestions,
        "word_count": len(cleaned),
        "char_count": len(text),
        "has_numbers": bool(re.search(r'\d', text)),
        "has_special_chars": bool(re.search(r'[^a-zA-Z0-9\s]', text)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MÉTRIQUES DE QUALITÉ
# ─────────────────────────────────────────────────────────────────────────────
def compute_readability_grade(avg_confidence: float, dict_hit_rate: float) -> tuple[str, str]:
    """
    Calcule un grade de lisibilité composite basé sur la confiance OCR
    et le taux de mots valides.
    
    Returns:
        (grade, description)
        Grade: A (excellent) à F (illisible)
    """
    # Score composite pondéré : 60% confiance OCR, 40% validité dictionnaire
    composite = (avg_confidence * 0.6) + (dict_hit_rate * 0.4)
    
    if composite >= 0.85:
        return "A", "Excellent — texte clair et bien reconnu"
    elif composite >= 0.70:
        return "B", "Bon — texte largement lisible, quelques incertitudes"
    elif composite >= 0.55:
        return "C", "Moyen — texte partiellement reconnu, vérification recommandée"
    elif composite >= 0.40:
        return "D", "Faible — texte difficile à lire, beaucoup d'incertitudes"
    else:
        return "F", "Illisible — reconnaissance très incertaine"


def compute_error_rates(hypothesis: str, reference: str) -> dict | None:
    """
    Calcule CER (Character Error Rate) et WER (Word Error Rate)
    en comparant le texte reconnu à une vérité terrain.
    
    Nécessite la bibliothèque jiwer.
    """
    if not reference or not hypothesis:
        return None
    
    if JIWER_AVAILABLE:
        try:
            character_error = cer(reference, hypothesis)
            word_error = wer(reference, hypothesis)
            return {
                "cer": round(character_error, 4),
                "wer": round(word_error, 4),
                "cer_percent": f"{character_error * 100:.1f}%",
                "wer_percent": f"{word_error * 100:.1f}%",
            }
        except Exception as e:
            print(f"   ⚠️ Erreur calcul CER/WER: {e}")
            return None
    else:
        # Implémentation de secours avec Levenshtein maison
        return _manual_error_rates(hypothesis, reference)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calcule la distance de Levenshtein entre deux chaînes."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, suppression, substitution
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def _manual_error_rates(hypothesis: str, reference: str) -> dict:
    """Calcul CER/WER sans jiwer (fallback)."""
    # CER — au niveau caractère
    char_dist = _levenshtein_distance(hypothesis.lower(), reference.lower())
    cer_val = char_dist / max(len(reference), 1)
    
    # WER — au niveau mot
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()
    word_dist = _levenshtein_distance(" ".join(hyp_words), " ".join(ref_words))
    # Approximation simple : distance sur la chaîne jointe / nb mots ref
    wer_val = word_dist / max(len(reference), 1)
    
    # Meilleure approche WER : Levenshtein sur les listes de mots
    word_lev = 0
    max_len = max(len(hyp_words), len(ref_words))
    for i in range(max_len):
        h = hyp_words[i] if i < len(hyp_words) else ""
        r = ref_words[i] if i < len(ref_words) else ""
        if h != r:
            word_lev += 1
    # Ajouter les mots en trop
    word_lev += abs(len(hyp_words) - len(ref_words))
    wer_val = word_lev / max(len(ref_words), 1)
    
    return {
        "cer": round(cer_val, 4),
        "wer": round(wer_val, 4),
        "cer_percent": f"{cer_val * 100:.1f}%",
        "wer_percent": f"{wer_val * 100:.1f}%",
        "note": "Calcul approximatif (installer jiwer pour plus de précision)",
    }


def compute_cross_engine_agreement(text1: str, text2: str) -> dict:
    """
    Compare les résultats de deux moteurs OCR via SequenceMatcher.
    
    Returns:
        dict avec ratio de similarité et détails
    """
    if not text1 or not text2:
        return {"similarity": 0.0, "status": "Comparaison impossible"}
    
    # Normaliser pour la comparaison
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    
    ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
    
    if ratio >= 0.9:
        status = "✅ Forte concordance — résultats fiables"
    elif ratio >= 0.7:
        status = "⚠️ Concordance partielle — vérification recommandée"
    elif ratio >= 0.4:
        status = "🟡 Faible concordance — texte probablement difficile"
    else:
        status = "❌ Divergence — les moteurs ne s'accordent pas"
    
    return {
        "similarity": round(ratio, 4),
        "similarity_percent": f"{ratio * 100:.1f}%",
        "status": status,
    }


def compute_multi_engine_agreement(results: dict) -> dict:
    """
    Compare tous les moteurs OCR entre eux pour créer une matrice d'accord.
    
    Args:
        results: dict engine_name -> result_dict
    
    Returns:
        dict avec matrix, avg_agreement, best_pair
    """
    texts = {}
    for name, r in results.items():
        if r and isinstance(r, dict) and r.get("text") and not r.get("error"):
            texts[name] = r["text"]
    
    if len(texts) < 2:
        return {"matrix": {}, "avg_agreement": 0.0, "best_pair": None}
    
    # Build pairwise similarity matrix
    matrix = {}
    all_scores = []
    best_score = 0.0
    best_pair = None
    
    engines = list(texts.keys())
    for i, e1 in enumerate(engines):
        matrix[e1] = {}
        for j, e2 in enumerate(engines):
            if i == j:
                matrix[e1][e2] = 1.0
            elif j > i:
                sim = difflib.SequenceMatcher(
                    None, texts[e1].lower().strip(), texts[e2].lower().strip()
                ).ratio()
                sim = round(sim, 4)
                matrix[e1][e2] = sim
                if e2 not in matrix:
                    matrix[e2] = {}
                matrix[e2][e1] = sim
                all_scores.append(sim)
                if sim > best_score:
                    best_score = sim
                    best_pair = (e1, e2, sim)
    
    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return {
        "matrix": matrix,
        "avg_agreement": round(avg, 4),
        "best_pair": best_pair,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. AFFICHAGE — Dashboard de résultats (CLI)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_print(text: str):
    """Print avec fallback ASCII si l'encodage ne supporte pas Unicode."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remplacer les caractères non-ASCII
        ascii_text = text.encode("ascii", errors="replace").decode("ascii")
        print(ascii_text)


def print_header(title: str):
    """Affiche un en-tête formaté."""
    width = 70
    _safe_print("")
    _safe_print("=" * width)
    _safe_print(f"  {title}")
    _safe_print("=" * width)


def print_section(title: str):
    """Affiche un titre de section."""
    _safe_print(f"\n-- {title} {'-' * (60 - len(title))}")


def print_results(
    easyocr_result: dict,
    trocr_result: dict | None,
    gemini_result: dict | None,
    mistral_result: dict | None,
    nlp_result: dict,
    grade: tuple[str, str],
    error_rates: dict | None,
    multi_agreement: dict,
    ai_analysis: dict | None,
    verbose: bool = False,
):
    """Affiche le dashboard complet des résultats."""
    
    # ── Texte extrait ──
    print_header("📝 TEXTE EXTRAIT")
    
    print_section("EasyOCR (local)")
    if easyocr_result["text"]:
        print(f"  \"{easyocr_result['text']}\"")
    else:
        print("  (aucun texte détecté)")
    
    if trocr_result:
        print_section("TrOCR (HF API)")
        if trocr_result["text"]:
            print(f"  \"{trocr_result['text']}\"")
        else:
            print("  (aucun texte détecté)")
            
    if gemini_result:
        print_section("Gemini Vision")
        if gemini_result.get("text") and not gemini_result.get("error"):
            print(f"  \"{gemini_result['text']}\"")
        else:
            print(f"  {gemini_result.get('text', '(erreur)')}")
            
    if mistral_result:
        print_section("Mistral OCR")
        if mistral_result.get("text") and not mistral_result.get("error"):
            print(f"  \"{mistral_result['text']}\"")
        else:
            print(f"  {mistral_result.get('text', '(erreur)')}")
            
    if ai_analysis and not ai_analysis.get("error"):
        print_section("🤖 Texte probable (AI Analysis)")
        print(f"  \"{ai_analysis.get('best_text', '')}\"")
        if ai_analysis.get("corrections"):
            print("\n  Corrections:")
            for c in ai_analysis["corrections"]:
                print(f"  • {c}")
    
    # ── Métriques de qualité ──
    print_header("📊 MÉTRIQUES DE QUALITÉ")
    
    # Grade de lisibilité
    grade_letter, grade_desc = grade
    grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
    grade_icon = grade_colors.get(grade_letter, "⚪")
    print(f"\n  {grade_icon} Grade global : {grade_letter} — {grade_desc}")
    
    if ai_analysis and not ai_analysis.get("error"):
        trust = ai_analysis.get("trust_score", 0)
        print(f"  🤖 AI Trust Score : {trust*100:.0f}%")
    
    # Tableau des métriques
    print_section("Détails")
    metrics = [
        ("Confiance moyenne (EasyOCR)", f"{easyocr_result['avg_confidence']:.1%}"),
        ("Confiance minimale (EasyOCR)", f"{easyocr_result['min_confidence']:.1%}"),
        ("Taux dictionnaire (NLP)", f"{nlp_result['dictionary_hit_rate']:.1%}"),
        ("Nombre de mots", str(nlp_result['word_count'])),
        ("Nombre de caractères", str(nlp_result['char_count'])),
        ("Temps EasyOCR", f"{easyocr_result['elapsed_seconds']}s"),
    ]
    
    if gemini_result:
        metrics.append(("Temps Gemini", f"{gemini_result.get('elapsed_seconds', 0)}s"))
    if mistral_result:
        metrics.append(("Temps Mistral", f"{mistral_result.get('elapsed_seconds', 0)}s"))
    if trocr_result:
        metrics.append(("Temps TrOCR", f"{trocr_result['elapsed_seconds']}s"))
        metrics.append(("Tentatives TrOCR", str(trocr_result['attempts'])))
    
    for label, value in metrics:
        print(f"  {label:<35} {value:>10}")
    
    # Concordance inter-moteurs
    if multi_agreement and multi_agreement.get("best_pair"):
        print_section("Concordance inter-moteurs (Meilleure paire)")
        e1, e2, sim = multi_agreement["best_pair"]
        print(f"  Moteurs: {e1} ↔ {e2}")
        print(f"  Similarité : {sim*100:.1f}%")
    
    # CER / WER
    if error_rates:
        print_section("Taux d'erreur (vs vérité terrain)")
        print(f"  CER (Character Error Rate) : {error_rates['cer_percent']}")
        print(f"  WER (Word Error Rate)      : {error_rates['wer_percent']}")
        if "note" in error_rates:
            print(f"  ℹ️  {error_rates['note']}")
    
    # ── Validation NLP ──
    print_header("🔤 VALIDATION NLP")
    
    if nlp_result["unknown_words"]:
        print_section("Mots non reconnus dans le dictionnaire")
        for word in nlp_result["unknown_words"]:
            suggestions = nlp_result["suggestions"].get(word, [])
            if suggestions:
                print(f"  ❓ \"{word}\"  →  peut-être : {', '.join(suggestions)}")
            else:
                print(f"  ❓ \"{word}\"  →  (pas de suggestion)")
    else:
        print("\n  ✅ Tous les mots sont reconnus dans le dictionnaire")
    
    if nlp_result["has_numbers"]:
        print("\n  🔢 Le texte contient des chiffres")
    if nlp_result["has_special_chars"]:
        print("  🔣 Le texte contient des caractères spéciaux")
    
    # ── Mode verbeux : détails par mot ──
    if verbose and easyocr_result["words"]:
        print_header("🔬 DÉTAILS PAR SEGMENT (mode --verbose)")
        
        print(f"\n  {'#':<4} {'Texte':<30} {'Confiance':<12} {'Statut'}")
        print(f"  {'─'*4} {'─'*30} {'─'*12} {'─'*15}")
        
        for i, word in enumerate(easyocr_result["words"], 1):
            conf = word["confidence"]
            if conf >= 0.8:
                status = "🟢 Fiable"
            elif conf >= 0.5:
                status = "🟡 Incertain"
            else:
                status = "🔴 Douteux"
            
            text_display = word["text"][:28]
            print(f"  {i:<4} {text_display:<30} {conf:<12.1%} {status}")
    
    _safe_print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI — Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    """Configure et parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="🖊️ OCR Manuscrit — Extraction de texte manuscrit avec validation NLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python ocr.py photo.jpg                         # OCR basique
  python ocr.py photo.jpg --verbose                # Détails par mot
  python ocr.py photo.jpg --ground-truth "Hello"   # Avec CER/WER
  python ocr.py photo.jpg --lang en fr             # Multi-langue

Variables d'environnement:
  HF_API_TOKEN    Token Hugging Face (optionnel, active TrOCR)
  GEMINI_API_KEY  Clé API Google Gemini (active Gemini Vision OCR)
  MISTRAL_API_KEY Clé API Mistral (active Mistral OCR)
        """,
    )
    
    parser.add_argument(
        "image",
        type=str,
        help="Chemin vers l'image manuscrite (jpg, png, etc.)",
    )
    
    parser.add_argument(
        "--ground-truth", "-gt",
        type=str,
        default=None,
        help="Texte attendu pour calculer CER/WER",
    )
    
    parser.add_argument(
        "--lang", "-l",
        nargs="+",
        default=["en"],
        help="Langue(s) pour EasyOCR (défaut: en). Ex: --lang en fr",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher les détails par segment (confiance, bbox)",
    )
    
    parser.add_argument(
        "--no-trocr",
        action="store_true",
        help="Désactiver TrOCR même si HF_API_TOKEN est défini",
    )
    
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Désactiver le prétraitement d'image",
    )
    
    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # ── Vérifier que l'image existe ──
    image_path = args.image
    if not os.path.isfile(image_path):
        print(f"❌ Erreur: Le fichier '{image_path}' n'existe pas.")
        sys.exit(1)
    
    # Vérifier l'extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    ext = Path(image_path).suffix.lower()
    if ext not in valid_extensions:
        print(f"⚠️ Attention: Extension '{ext}' non standard. Extensions supportées: {', '.join(valid_extensions)}")
    
    print_header("🖊️ OCR MANUSCRIT — Analyse en cours")
    print(f"\n  📁 Image    : {image_path}")
    print(f"  📐 Taille   : {os.path.getsize(image_path) / 1024:.1f} Ko")
    print(f"  🌐 Langues  : {', '.join(args.lang)}")
    print(f"  🔑 HF Token : {'✅ Défini' if os.environ.get('HF_API_TOKEN') else '❌ Non défini'}")
    print(f"  ♊ Gemini   : {'✅ Défini' if os.environ.get('GEMINI_API_KEY') else '❌ Non défini'}")
    print(f"  🌀 Mistral  : {'✅ Défini' if os.environ.get('MISTRAL_API_KEY') else '❌ Non défini'}")
    
    # ── 1. Prétraitement ──
    if not args.no_preprocess:
        print_section("Prétraitement")
        try:
            preprocessed = preprocess_image(image_path)
            temp_path = str(Path(image_path).parent / f"_preprocessed_{Path(image_path).name}")
            cv2.imwrite(temp_path, preprocessed)
            ocr_image_path = temp_path
            print("  ✅ Image prétraitée (contraste + débruitage)")
        except Exception as e:
            print(f"  ⚠️ Prétraitement échoué ({e}), utilisation de l'original")
            ocr_image_path = image_path
            temp_path = None
    else:
        ocr_image_path = image_path
        temp_path = None
        print("\n  ⏭️ Prétraitement désactivé")
    
    # ── 2. EasyOCR ──
    print_section("Moteurs OCR")
    easyocr_result = ocr_easyocr(ocr_image_path, languages=args.lang)
    
    # ── 2.B Gemini Vision ──
    gemini_result = None
    if os.environ.get("GEMINI_API_KEY"):
        gemini_result = ocr_gemini(ocr_image_path)
        
    # ── 2.C Mistral OCR ──
    mistral_result = None
    if os.environ.get("MISTRAL_API_KEY"):
        mistral_result = ocr_mistral(ocr_image_path)
    
    # ── 3. TrOCR (optionnel) ──
    trocr_result = None
    if not args.no_trocr:
        trocr_result = ocr_trocr_api(image_path)
    
    # ── Nettoyage du fichier temporaire ──
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
    
    # ── 4. Texte principal (EasyOCR prioritaire car il a les scores) ──
    primary_text = easyocr_result["text"]
    
    # ── 5. Validation NLP ──
    nlp_result = validate_nlp(primary_text)
    
    # ── 6. Grade de lisibilité ──
    grade = compute_readability_grade(
        easyocr_result["avg_confidence"],
        nlp_result["dictionary_hit_rate"],
    )
    
    # ── 7. CER/WER (si ground truth fourni) ──
    error_rates = None
    if args.ground_truth:
        error_rates = compute_error_rates(primary_text, args.ground_truth)
    
    # ── 8. Concordance inter-moteurs ──
    all_results = {"EasyOCR": easyocr_result}
    if gemini_result and not gemini_result.get("error"):
        all_results["Gemini"] = gemini_result
    if mistral_result and not mistral_result.get("error"):
        all_results["Mistral"] = mistral_result
    if trocr_result:
        all_results["TrOCR"] = trocr_result
        
    multi_agreement = compute_multi_engine_agreement(all_results)
    
    # ── 8.B AI Analysis (si Gemini présent) ──
    ai_analysis = None
    if os.environ.get("GEMINI_API_KEY") and len(all_results) > 1:
        ai_analysis = analyze_with_ai(all_results, image_path=ocr_image_path)
    
    # ── 9. Affichage du dashboard ──
    print_results(
        easyocr_result=easyocr_result,
        trocr_result=trocr_result,
        gemini_result=gemini_result,
        mistral_result=mistral_result,
        nlp_result=nlp_result,
        grade=grade,
        error_rates=error_rates,
        multi_agreement=multi_agreement,
        ai_analysis=ai_analysis,
        verbose=args.verbose,
    )
    
    # Retourner le texte pour usage programmatique
    return primary_text


if __name__ == "__main__":
    main()
