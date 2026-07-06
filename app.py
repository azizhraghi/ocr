#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  OCR Manuscrit — Dashboard Streamlit v2.0
=============================================================================

Dashboard interactif pour tester l'OCR manuscrit avec:
- Triple moteur: EasyOCR + Gemini Vision + Mistral OCR
- AI Analysis: Gemini-powered intelligent feedback
- n8n Webhook integration for automation
- Engine competition leaderboard

Usage:
    streamlit run app.py
"""

import io
import os
import sys
import tempfile
import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

# Importer les fonctions du module OCR principal
from ocr import (
    preprocess_image,
    ocr_easyocr,
    ocr_gemini,
    ocr_mistral,
    ocr_trocr_api,
    validate_nlp,
    compute_readability_grade,
    compute_error_rates,
    compute_cross_engine_agreement,
    compute_multi_engine_agreement,
    analyze_with_ai,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Manuscrit — AI Pipeline",
    page_icon="🖊️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS personnalisé — Design premium dark
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Polices Google ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Variables globales ── */
    :root {
        --primary: #6C63FF;
        --primary-light: #8B83FF;
        --secondary: #00D4AA;
        --accent: #FF6B9D;
        --bg-dark: #0E1117;
        --bg-card: #1A1D29;
        --bg-card-hover: #222639;
        --text-primary: #FAFAFA;
        --text-secondary: #A0A4B8;
        --border: #2D3148;
        --success: #00D4AA;
        --warning: #FFB84D;
        --danger: #FF5C5C;
        --info: #5BA4FF;
        --gemini: #4285F4;
        --mistral: #FF7000;
    }

    /* ── Corps principal ── */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* ── Carte de métrique ── */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-hover) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 16px 16px 0 0;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--primary);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 8px 0;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 4px;
    }

    /* ── Grade badge ── */
    .grade-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: white;
        margin: 0 auto 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .grade-A { background: linear-gradient(135deg, #00D4AA, #00B894); }
    .grade-B { background: linear-gradient(135deg, #5BA4FF, #3D7BFF); }
    .grade-C { background: linear-gradient(135deg, #FFB84D, #FF9F1C); }
    .grade-D { background: linear-gradient(135deg, #FF8C42, #FF6B35); }
    .grade-F { background: linear-gradient(135deg, #FF5C5C, #E63946); }

    /* ── Texte extrait ── */
    .extracted-text {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        font-size: 1.15rem;
        line-height: 1.8;
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
        margin: 12px 0;
    }

    /* ── Section header ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border);
        color: var(--text-primary);
    }

    /* ── Word badge ── */
    .word-known {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        color: var(--success);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 8px;
        padding: 4px 12px;
        margin: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .word-unknown {
        display: inline-block;
        background: rgba(255, 92, 92, 0.15);
        color: var(--danger);
        border: 1px solid rgba(255, 92, 92, 0.3);
        border-radius: 8px;
        padding: 4px 12px;
        margin: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    /* ── Confidence bar ── */
    .conf-bar-bg {
        background: var(--bg-card);
        border-radius: 6px;
        height: 10px;
        width: 100%;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-card);
    }

    /* ── Hide Streamlit default ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Engine tag ── */
    .engine-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(108, 99, 255, 0.12);
        color: var(--primary-light);
        border: 1px solid rgba(108, 99, 255, 0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── AI Analysis card ── */
    .ai-analysis-card {
        background: linear-gradient(135deg, #1a1d29 0%, #1e2235 100%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 28px;
        margin: 16px 0;
        position: relative;
        overflow: hidden;
    }
    .ai-analysis-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6C63FF, #FF6B9D, #00D4AA);
    }

    /* ── Leaderboard ── */
    .leaderboard-item {
        display: flex;
        align-items: center;
        gap: 16px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .leaderboard-item:hover {
        border-color: var(--primary);
        transform: translateX(4px);
    }
    .leaderboard-rank {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        min-width: 50px;
        text-align: center;
    }
    .leaderboard-engine {
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--text-primary);
    }
    .leaderboard-score {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.2rem;
        margin-left: auto;
    }
    .leaderboard-reason {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* ── Smooth animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def render_metric_card(icon: str, value: str, label: str, color: str = "#6C63FF"):
    """Render une carte de métrique stylée."""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_grade_badge(grade: str):
    """Render le badge de grade circulaire."""
    return f"""
    <div style="text-align: center;">
        <div class="grade-badge grade-{grade}">{grade}</div>
    </div>
    """


def get_confidence_color(conf: float) -> str:
    """Retourne une couleur basée sur la confiance."""
    if conf >= 0.8:
        return "#00D4AA"
    elif conf >= 0.6:
        return "#5BA4FF"
    elif conf >= 0.4:
        return "#FFB84D"
    else:
        return "#FF5C5C"


def save_uploaded_to_temp(uploaded_file) -> str:
    """Sauvegarde un fichier uploadé dans un fichier temporaire et retourne le chemin."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def render_leaderboard(engine_ranking: list):
    """Render the engine competition leaderboard."""
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#A0A4B8", "#A0A4B8"]

    html = ""
    for i, entry in enumerate(engine_ranking):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        color = colors[i] if i < len(colors) else "#A0A4B8"
        engine = entry.get("engine", "?")
        score = entry.get("score", 0)
        reason = entry.get("reason", "")
        score_pct = f"{score * 100:.0f}%"
        
        html += f"""
        <div class="leaderboard-item">
            <div class="leaderboard-rank" style="color: {color};">{medal}</div>
            <div>
                <div class="leaderboard-engine">{engine}</div>
                <div class="leaderboard-reason">{reason}</div>
            </div>
            <div class="leaderboard-score" style="color: {get_confidence_color(score)};">{score_pct}</div>
        </div>
        """
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Langues
    languages = st.multiselect(
        "🌐 Langues OCR",
        options=["en", "fr", "es", "de", "it", "pt", "ar", "zh", "ja", "ko"],
        default=["en"],
        help="Langues pour le moteur EasyOCR"
    )
    if not languages:
        languages = ["en"]

    st.markdown("---")
    st.markdown("### 🔧 Options")

    # Prétraitement
    enable_preprocess = st.toggle("🔧 Prétraitement d'image", value=True,
                                   help="Améliore le contraste et réduit le bruit")

    st.markdown("---")
    st.markdown("### 🏆 Moteurs OCR")

    # Gemini
    enable_gemini = st.toggle("♊ Gemini Vision OCR", value=True,
                               help="Google Gemini AI — extraction haute qualité")

    # Mistral
    enable_mistral = st.toggle("🌀 Mistral OCR", value=True,
                                help="Mistral AI — OCR spécialisé documents")

    # TrOCR
    enable_trocr = st.toggle("🌐 TrOCR (HF API)", value=False,
                              help="Nécessite HF_API_TOKEN")

    # AI Analysis
    st.markdown("---")
    st.markdown("### 🤖 Intelligence")
    enable_ai_analysis = st.toggle("🤖 AI Analysis", value=True,
                                    help="Analyse intelligente par Gemini des résultats OCR")

    st.markdown("---")
    st.markdown("### 🔑 Clés API")

    gemini_key = st.text_input("♊ Gemini API Key", type="password",
                                value=os.environ.get("GEMINI_API_KEY", ""),
                                help="Clé API Google Gemini")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    mistral_key = st.text_input("🌀 Mistral API Key", type="password",
                                 value=os.environ.get("MISTRAL_API_KEY", ""),
                                 help="Clé API Mistral AI")
    if mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key

    if enable_trocr:
        hf_token = st.text_input("🔑 HF API Token", type="password",
                                  value=os.environ.get("HF_API_TOKEN", ""),
                                  help="Token Hugging Face pour l'API d'inférence")
        if hf_token:
            os.environ["HF_API_TOKEN"] = hf_token

    st.markdown("---")

    # n8n webhook
    st.markdown("### 🔗 Automation")
    n8n_webhook_url = st.text_input(
        "n8n Webhook URL",
        value="",
        placeholder="https://your-n8n.app/webhook/...",
        help="URL du webhook n8n pour l'automatisation OCR"
    )

    st.markdown("---")

    # Ground truth
    st.markdown("### 📏 Vérité terrain")
    ground_truth = st.text_area(
        "Texte attendu (optionnel)",
        placeholder="Entrez le texte que l'image devrait contenir...",
        help="Pour calculer CER/WER — laissez vide si inconnu"
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.75rem;'>"
        "OCR Manuscrit v2.0<br>"
        "EasyOCR + Gemini + Mistral + AI"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-bottom: 8px;">
    <h1 style="font-size: 2.5rem; font-weight: 800; margin-bottom: 0;">
        🖊️ OCR Manuscrit
    </h1>
    <p style="color: #A0A4B8; font-size: 1.1rem; margin-top: 4px;">
        Extraction de texte manuscrit avec triple moteur OCR, analyse IA et métriques de qualité
    </p>
</div>
""", unsafe_allow_html=True)

# ── Upload zone ──
uploaded_file = st.file_uploader(
    "Déposez votre image manuscrite",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
    help="Formats supportés: JPG, PNG, BMP, TIFF, WebP"
)

if uploaded_file is not None:
    # ── Afficher l'image ──
    image = Image.open(uploaded_file)

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)

    with col_info:
        st.markdown("### 📋 Informations")
        st.markdown(f"**Fichier:** {uploaded_file.name}")
        st.markdown(f"**Taille:** {uploaded_file.size / 1024:.1f} Ko")
        st.markdown(f"**Dimensions:** {image.size[0]} × {image.size[1]} px")
        st.markdown(f"**Mode:** {image.mode}")
        st.markdown(f"**Langues:** {', '.join(languages)}")

        engines_status = []
        engines_status.append("✅ EasyOCR (toujours actif)")
        if enable_gemini:
            engines_status.append("✅ Gemini Vision" if gemini_key else "⚠️ Gemini (clé manquante)")
        if enable_mistral:
            engines_status.append("✅ Mistral OCR" if mistral_key else "⚠️ Mistral (clé manquante)")
        if enable_trocr:
            engines_status.append("✅ TrOCR")

        st.markdown("**Moteurs actifs:**")
        for s in engines_status:
            st.markdown(f"  {s}")

    st.markdown("---")

    # ── Bouton d'analyse ──
    if st.button("🚀 Lancer l'analyse OCR", type="primary", use_container_width=True):

        # Sauvegarder dans un fichier temporaire
        temp_path = save_uploaded_to_temp(uploaded_file)

        try:
            # ── 1. Prétraitement ──
            prep_path = None
            if enable_preprocess:
                with st.spinner("🔧 Prétraitement de l'image..."):
                    try:
                        preprocessed = preprocess_image(temp_path)
                        import cv2
                        prep_path = temp_path + "_prep.png"
                        cv2.imwrite(prep_path, preprocessed)

                        # Montrer avant/après
                        with st.expander("🔍 Avant / Après prétraitement", expanded=False):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(image, caption="Original", use_container_width=True)
                            with c2:
                                st.image(prep_path, caption="Prétraitée", use_container_width=True)

                        ocr_path = prep_path
                    except Exception as e:
                        st.warning(f"⚠️ Prétraitement échoué: {e}. Utilisation de l'original.")
                        ocr_path = temp_path
                        prep_path = None
            else:
                ocr_path = temp_path

            # ── 2. EasyOCR ──
            with st.spinner("🔍 EasyOCR — Analyse en cours..."):
                easyocr_result = ocr_easyocr(ocr_path, languages=languages)

            # ── 3. Gemini Vision OCR ──
            gemini_result = None
            if enable_gemini and gemini_key:
                with st.spinner("♊ Gemini Vision — Analyse en cours..."):
                    gemini_result = ocr_gemini(temp_path, api_key=gemini_key)

            # ── 4. Mistral OCR ──
            mistral_result = None
            if enable_mistral and mistral_key:
                with st.spinner("🌀 Mistral OCR — Analyse en cours..."):
                    mistral_result = ocr_mistral(temp_path, api_key=mistral_key)

            # ── 5. TrOCR (optionnel) ──
            trocr_result = None
            if enable_trocr:
                with st.spinner("🌐 TrOCR (HF API) — Envoi en cours..."):
                    trocr_result = ocr_trocr_api(temp_path)

            # ── 6. n8n Webhook (optionnel) ──
            n8n_result = None
            if n8n_webhook_url:
                with st.spinner("🔗 n8n Webhook — Envoi en cours..."):
                    try:
                        import requests
                        with open(temp_path, "rb") as f:
                            files = {"image": (uploaded_file.name, f, "image/png")}
                            resp = requests.post(n8n_webhook_url, files=files, timeout=60)
                        if resp.status_code == 200:
                            n8n_result = resp.json()
                        else:
                            st.warning(f"⚠️ n8n webhook returned {resp.status_code}")
                    except Exception as e:
                        st.warning(f"⚠️ n8n webhook failed: {e}")

            # ── 7. NLP Validation ──
            primary_text = easyocr_result["text"]
            nlp_result = validate_nlp(primary_text)

            # ── 8. Grade ──
            grade_letter, grade_desc = compute_readability_grade(
                easyocr_result["avg_confidence"],
                nlp_result["dictionary_hit_rate"],
            )

            # ── 9. Error rates ──
            error_rates = None
            if ground_truth.strip():
                error_rates = compute_error_rates(primary_text, ground_truth.strip())

            # ── 10. Multi-engine agreement ──
            all_results = {"EasyOCR": easyocr_result}
            if gemini_result and not gemini_result.get("error"):
                all_results["Gemini"] = gemini_result
            if mistral_result and not mistral_result.get("error"):
                all_results["Mistral"] = mistral_result
            if trocr_result:
                all_results["TrOCR"] = trocr_result

            multi_agreement = compute_multi_engine_agreement(all_results)

            # ── 11. AI Analysis ──
            ai_analysis = None
            if enable_ai_analysis and gemini_key:
                with st.spinner("🤖 AI Analysis — Analyse intelligente en cours..."):
                    ai_analysis = analyze_with_ai(
                        all_results,
                        image_path=temp_path,
                        api_key=gemini_key,
                    )
            
            # ── 12. Save History ──
            history_file = "ocr_history.json"
            try:
                history = []
                if os.path.exists(history_file):
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                
                # Append current result
                history.append({
                    "timestamp": datetime.now().isoformat(),
                    "filename": uploaded_file.name,
                    "grade": grade_letter,
                    "avg_confidence": easyocr_result["avg_confidence"],
                    "trust_score": ai_analysis.get("trust_score", 0) if ai_analysis and not ai_analysis.get("error") else 0,
                    "engines_used": [eng for eng, res in all_results.items() if res and not res.get("error")],
                    "text": ai_analysis.get("best_text", primary_text) if ai_analysis and not ai_analysis.get("error") else primary_text
                })
                
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                st.warning(f"Impossible de sauvegarder l'historique: {e}")

            # Store in session state
            st.session_state["results"] = {
                "easyocr": easyocr_result,
                "gemini": gemini_result,
                "mistral": mistral_result,
                "trocr": trocr_result,
                "n8n": n8n_result,
                "nlp": nlp_result,
                "grade": (grade_letter, grade_desc),
                "error_rates": error_rates,
                "multi_agreement": multi_agreement,
                "ai_analysis": ai_analysis,
            }

        finally:
            # Nettoyage
            try:
                os.unlink(temp_path)
                if prep_path and os.path.exists(prep_path):
                    os.unlink(prep_path)
            except OSError:
                pass

    # ── Afficher les résultats ──
    if "results" in st.session_state:
        results = st.session_state["results"]
        easyocr_result = results["easyocr"]
        gemini_result = results.get("gemini")
        mistral_result = results.get("mistral")
        trocr_result = results.get("trocr")
        n8n_result = results.get("n8n")
        nlp_result = results["nlp"]
        grade_letter, grade_desc = results["grade"]
        error_rates = results["error_rates"]
        multi_agreement = results.get("multi_agreement", {})
        ai_analysis = results.get("ai_analysis")

        # ════════════════════════════════════════════════════════════
        # SECTION 0: AI Analysis (featured at the top)
        # ════════════════════════════════════════════════════════════
        if ai_analysis and not ai_analysis.get("error"):
            st.markdown("""
            <div class="section-header">🤖 AI Analysis — Interprétation Intelligente</div>
            """, unsafe_allow_html=True)

            # Best text
            best_text = ai_analysis.get("best_text", "")
            trust_score = ai_analysis.get("trust_score", 0.0)
            interpretation = ai_analysis.get("interpretation", "")

            trust_color = get_confidence_color(trust_score)

            st.markdown(f"""
            <div class="ai-analysis-card">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <span style="font-size: 1.5rem;">🤖</span>
                    <span style="font-weight: 700; font-size: 1.1rem; color: var(--text-primary);">Texte le plus probable</span>
                    <span style="margin-left: auto; font-family: 'JetBrains Mono'; font-weight: 700; font-size: 1.3rem; color: {trust_color};">
                        {trust_score * 100:.0f}% confiance
                    </span>
                </div>
                <div style="background: rgba(108,99,255,0.08); border-radius: 12px; padding: 16px 20px; font-size: 1.2rem; line-height: 1.6; color: var(--text-primary); border: 1px solid rgba(108,99,255,0.2);">
                    "{best_text}"
                </div>
                <p style="margin-top: 16px; color: var(--text-secondary); font-size: 0.95rem; line-height: 1.6;">
                    {interpretation}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Corrections
            corrections = ai_analysis.get("corrections", [])
            if corrections:
                with st.expander("✏️ Corrections suggérées par l'IA", expanded=True):
                    for c in corrections:
                        st.markdown(f"• {c}")

            # Engine Leaderboard
            engine_ranking = ai_analysis.get("engine_ranking", [])
            if engine_ranking:
                st.markdown("""
                <div class="section-header">🏆 Classement des moteurs — Competition</div>
                """, unsafe_allow_html=True)

                st.markdown(render_leaderboard(engine_ranking), unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════
        # SECTION 1: Texte extrait par moteur
        # ════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-header">📝 Texte extrait par moteur</div>
        """, unsafe_allow_html=True)

        if easyocr_result["text"]:
            st.markdown(f"""
            <div class="extracted-text">
                <span class="engine-tag">🔍 EasyOCR</span>
                <p style="margin-top: 12px; margin-bottom: 0;">{easyocr_result['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Aucun texte détecté par EasyOCR")

        if gemini_result and gemini_result.get("text") and not gemini_result.get("error"):
            st.markdown(f"""
            <div class="extracted-text" style="border-left-color: #4285F4; margin-top: 12px;">
                <span class="engine-tag" style="background: rgba(66,133,244,0.12); color: #4285F4; border-color: rgba(66,133,244,0.25);">♊ Gemini Vision</span>
                <p style="margin-top: 12px; margin-bottom: 0;">{gemini_result['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif gemini_result and gemini_result.get("error"):
            st.error(f"Gemini: {gemini_result.get('text', 'Erreur')}")

        if mistral_result and mistral_result.get("text") and not mistral_result.get("error"):
            st.markdown(f"""
            <div class="extracted-text" style="border-left-color: #FF7000; margin-top: 12px;">
                <span class="engine-tag" style="background: rgba(255,112,0,0.12); color: #FF7000; border-color: rgba(255,112,0,0.25);">🌀 Mistral OCR</span>
                <p style="margin-top: 12px; margin-bottom: 0;">{mistral_result['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif mistral_result and mistral_result.get("error"):
            st.error(f"Mistral: {mistral_result.get('text', 'Erreur')}")

        if trocr_result and trocr_result.get("text"):
            st.markdown(f"""
            <div class="extracted-text" style="border-left-color: #00D4AA; margin-top: 12px;">
                <span class="engine-tag" style="background: rgba(0,212,170,0.12); color: #00D4AA; border-color: rgba(0,212,170,0.25);">🌐 TrOCR</span>
                <p style="margin-top: 12px; margin-bottom: 0;">{trocr_result['text']}</p>
            </div>
            """, unsafe_allow_html=True)

        if n8n_result:
            st.markdown(f"""
            <div class="extracted-text" style="border-left-color: #FF6D5A; margin-top: 12px;">
                <span class="engine-tag" style="background: rgba(255,109,90,0.12); color: #FF6D5A; border-color: rgba(255,109,90,0.25);">🔗 n8n Workflow</span>
                <p style="margin-top: 12px; margin-bottom: 0;">{n8n_result}</p>
            </div>
            """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════
        # SECTION 2: Métriques de qualité
        # ════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-header">📊 Métriques de qualité</div>
        """, unsafe_allow_html=True)

        # Grade + Métriques en colonnes
        col_grade, col_m1, col_m2, col_m3, col_m4 = st.columns([1.2, 1, 1, 1, 1])

        with col_grade:
            st.markdown(render_grade_badge(grade_letter), unsafe_allow_html=True)
            st.markdown(
                f"<div style='text-align:center; color: #A0A4B8; font-size: 0.85rem;'>{grade_desc}</div>",
                unsafe_allow_html=True
            )

        with col_m1:
            conf_color = get_confidence_color(easyocr_result["avg_confidence"])
            st.markdown(render_metric_card(
                "🎯", f"{easyocr_result['avg_confidence']:.0%}",
                "Confiance EasyOCR", conf_color
            ), unsafe_allow_html=True)

        with col_m2:
            dict_color = get_confidence_color(nlp_result["dictionary_hit_rate"])
            st.markdown(render_metric_card(
                "📖", f"{nlp_result['dictionary_hit_rate']:.0%}",
                "Taux dictionnaire", dict_color
            ), unsafe_allow_html=True)

        with col_m3:
            st.markdown(render_metric_card(
                "🔢", str(nlp_result["word_count"]),
                "Mots détectés", "#8B83FF"
            ), unsafe_allow_html=True)

        with col_m4:
            # AI trust score if available
            if ai_analysis and not ai_analysis.get("error"):
                trust = ai_analysis.get("trust_score", 0)
                trust_col = get_confidence_color(trust)
                st.markdown(render_metric_card(
                    "🤖", f"{trust * 100:.0f}%",
                    "Trust Score (AI)", trust_col
                ), unsafe_allow_html=True)
            else:
                min_color = get_confidence_color(easyocr_result["min_confidence"])
                st.markdown(render_metric_card(
                    "📉", f"{easyocr_result['min_confidence']:.0%}",
                    "Confiance min", min_color
                ), unsafe_allow_html=True)

        # ── Ligne 2: Temps par moteur ──
        time_cols_data = [("🔍", f"{easyocr_result['elapsed_seconds']}s", "Temps EasyOCR")]

        if gemini_result and not gemini_result.get("error"):
            time_cols_data.append(("♊", f"{gemini_result['elapsed_seconds']}s", "Temps Gemini"))
        if mistral_result and not mistral_result.get("error"):
            time_cols_data.append(("🌀", f"{mistral_result['elapsed_seconds']}s", "Temps Mistral"))
        if trocr_result:
            time_cols_data.append(("🌐", f"{trocr_result['elapsed_seconds']}s", "Temps TrOCR"))
        if error_rates:
            time_cols_data.append(("CER", error_rates["cer_percent"], "Character Error Rate"))
            time_cols_data.append(("WER", error_rates["wer_percent"], "Word Error Rate"))
        if multi_agreement.get("avg_agreement"):
            time_cols_data.append(("🤝", f"{multi_agreement['avg_agreement'] * 100:.0f}%", "Concordance moyenne"))

        if time_cols_data:
            cols = st.columns(len(time_cols_data))
            for i, (icon, val, label) in enumerate(time_cols_data):
                with cols[i]:
                    st.markdown(render_metric_card(icon, val, label), unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════
        # SECTION 3: Concordance inter-moteurs
        # ════════════════════════════════════════════════════════════
        if multi_agreement.get("matrix"):
            st.markdown("""
            <div class="section-header">🤝 Concordance inter-moteurs</div>
            """, unsafe_allow_html=True)

            if multi_agreement.get("best_pair"):
                e1, e2, sim = multi_agreement["best_pair"]
                if sim >= 0.9:
                    status = "✅ Forte concordance"
                elif sim >= 0.7:
                    status = "⚠️ Concordance partielle"
                else:
                    status = "🟡 Faible concordance"
                st.markdown(f"**Meilleure paire:** {e1} ↔ {e2} — **{sim * 100:.0f}%** {status}")

            # Show matrix as table
            import pandas as pd
            matrix = multi_agreement["matrix"]
            engines = list(matrix.keys())
            df_data = {}
            for e1 in engines:
                row = {}
                for e2 in engines:
                    val = matrix.get(e1, {}).get(e2, 0)
                    row[e2] = f"{val * 100:.0f}%"
                df_data[e1] = row
            df = pd.DataFrame(df_data).T
            st.dataframe(df, use_container_width=True)

        # ════════════════════════════════════════════════════════════
        # SECTION 4: Validation NLP
        # ════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-header">🔤 Validation NLP</div>
        """, unsafe_allow_html=True)

        # Visual word map
        if nlp_result["word_count"] > 0:
            all_words = nlp_result["known_words"] + nlp_result["unknown_words"]
            known_set = set(nlp_result["known_words"])

            word_html = '<div style="display: flex; flex-wrap: wrap; gap: 4px; margin: 12px 0;">'
            for w in all_words:
                if w in known_set:
                    word_html += f'<span class="word-known">{w}</span>'
                else:
                    word_html += f'<span class="word-unknown">{w} ❓</span>'
            word_html += '</div>'
            st.markdown(word_html, unsafe_allow_html=True)

        # Suggestions
        if nlp_result["suggestions"]:
            with st.expander("💡 Suggestions de correction", expanded=True):
                for word, suggestions in nlp_result["suggestions"].items():
                    st.markdown(f"**\"{word}\"** → peut-être : _{', '.join(suggestions)}_")

        if nlp_result["has_numbers"]:
            st.info("🔢 Le texte contient des chiffres")
        if nlp_result["has_special_chars"]:
            st.info("🔣 Le texte contient des caractères spéciaux")

        # ════════════════════════════════════════════════════════════
        # SECTION 5: Détails par segment
        # ════════════════════════════════════════════════════════════
        if easyocr_result["words"]:
            with st.expander("🔬 Détails par segment (confiance)", expanded=False):
                import pandas as pd

                rows = []
                for i, w in enumerate(easyocr_result["words"], 1):
                    conf = w["confidence"]
                    if conf >= 0.8:
                        status = "✅ Fiable"
                    elif conf >= 0.5:
                        status = "⚠️ Incertain"
                    else:
                        status = "❌ Douteux"

                    rows.append({
                        "#": i,
                        "Texte": w["text"],
                        "Confiance": f"{conf:.1%}",
                        "Statut": status,
                    })

                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "#": st.column_config.NumberColumn(width="small"),
                        "Texte": st.column_config.TextColumn(width="large"),
                        "Confiance": st.column_config.TextColumn(width="medium"),
                        "Statut": st.column_config.TextColumn(width="medium"),
                    }
                )

        # ════════════════════════════════════════════════════════════
        # SECTION 6: CER/WER détails
        # ════════════════════════════════════════════════════════════
        if error_rates:
            with st.expander("📏 Taux d'erreur détaillés (CER / WER)", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    cer_val = error_rates["cer"]
                    st.metric("Character Error Rate (CER)", error_rates["cer_percent"],
                              delta=None if cer_val < 0.1 else "Élevé",
                              delta_color="off" if cer_val < 0.1 else "inverse")
                with c2:
                    wer_val = error_rates["wer"]
                    st.metric("Word Error Rate (WER)", error_rates["wer_percent"],
                              delta=None if wer_val < 0.15 else "Élevé",
                              delta_color="off" if wer_val < 0.15 else "inverse")

                if "note" in error_rates:
                    st.caption(f"ℹ️ {error_rates['note']}")

else:
    # ── État vide — Instructions ──
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; animation: fadeInUp 0.6s ease forwards;">
        <div style="font-size: 4rem; margin-bottom: 20px;">📷</div>
        <h3 style="color: #A0A4B8; font-weight: 400; margin-bottom: 24px;">
            Déposez une image manuscrite pour commencer
        </h3>
        <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🏆</div>
                <p style="color: #666; font-size: 0.9rem;">Triple moteur OCR<br><small>EasyOCR + Gemini + Mistral</small></p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🤖</div>
                <p style="color: #666; font-size: 0.9rem;">AI Analysis<br><small>Gemini intelligent feedback</small></p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">📊</div>
                <p style="color: #666; font-size: 0.9rem;">Métriques de qualité<br><small>Confiance, CER, WER</small></p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🔗</div>
                <p style="color: #666; font-size: 0.9rem;">Automation<br><small>n8n Webhook</small></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
