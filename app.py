#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  OCR Manuscrit — Dashboard Streamlit
=============================================================================

Dashboard interactif pour tester l'OCR manuscrit avec visualisation
des métriques, validation NLP, et comparaison de moteurs.

Usage:
    streamlit run app.py
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

# Importer les fonctions du module OCR principal
from ocr import (
    preprocess_image,
    ocr_easyocr,
    ocr_paddleocr,
    ocr_trocr_api,
    validate_nlp,
    compute_readability_grade,
    compute_error_rates,
    compute_cross_engine_agreement,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Manuscrit",
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

    # Prétraitement
    enable_preprocess = st.toggle("🔧 Prétraitement d'image", value=True,
                                   help="Améliore le contraste et réduit le bruit")

    # PaddleOCR
    enable_paddleocr = st.toggle("🚀 PaddleOCR (Local)", value=True,
                                  help="Moteur local haute performance")

    # TrOCR
    enable_trocr = st.toggle("🌐 TrOCR (HF API)", value=False,
                              help="Nécessite HF_API_TOKEN")

    if enable_trocr:
        hf_token = st.text_input("🔑 HF API Token", type="password",
                                  value=os.environ.get("HF_API_TOKEN", ""),
                                  help="Token Hugging Face pour l'API d'inférence")
        if hf_token:
            os.environ["HF_API_TOKEN"] = hf_token

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
        "OCR Manuscrit v1.0<br>"
        "EasyOCR + TrOCR + NLP"
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
        Extraction de texte manuscrit avec validation NLP et métriques de qualité
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
        st.markdown(f"**Prétraitement:** {'✅' if enable_preprocess else '❌'}")
        st.markdown(f"**PaddleOCR:** {'✅' if enable_paddleocr else '❌'}")
        st.markdown(f"**TrOCR:** {'✅' if enable_trocr else '❌'}")

    st.markdown("---")

    # ── Bouton d'analyse ──
    if st.button("🚀 Lancer l'analyse OCR", type="primary", use_container_width=True):

        # Sauvegarder dans un fichier temporaire
        temp_path = save_uploaded_to_temp(uploaded_file)

        try:
            # ── 1. Prétraitement ──
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
                prep_path = None

            # ── 2. EasyOCR ──
            with st.spinner("🔍 EasyOCR — Analyse en cours..."):
                easyocr_result = ocr_easyocr(ocr_path, languages=languages)

            # ── 3. PaddleOCR (optionnel) ──
            paddleocr_result = None
            if enable_paddleocr:
                with st.spinner("🚀 PaddleOCR — Analyse en cours..."):
                    paddleocr_result = ocr_paddleocr(temp_path, languages=languages)

            # ── 4. TrOCR (optionnel) ──
            trocr_result = None
            if enable_trocr:
                with st.spinner("🌐 TrOCR (HF API) — Envoi en cours..."):
                    trocr_result = ocr_trocr_api(temp_path)

            # ── 5. NLP Validation ──
            primary_text = easyocr_result["text"]
            nlp_result = validate_nlp(primary_text)

            # ── 6. Grade ──
            grade_letter, grade_desc = compute_readability_grade(
                easyocr_result["avg_confidence"],
                nlp_result["dictionary_hit_rate"],
            )

            # ── 7. Error rates ──
            error_rates = None
            if ground_truth.strip():
                error_rates = compute_error_rates(primary_text, ground_truth.strip())

            # ── 8. Cross-engine agreement ──
            cross_agreement = None
            if paddleocr_result and paddleocr_result.get("text"):
                cross_agreement = compute_cross_engine_agreement(
                    easyocr_result["text"], paddleocr_result["text"]
                )
            elif trocr_result and trocr_result.get("text"):
                cross_agreement = compute_cross_engine_agreement(
                    easyocr_result["text"], trocr_result["text"]
                )

            # Store in session state
            st.session_state["results"] = {
                "easyocr": easyocr_result,
                "paddleocr": paddleocr_result,
                "trocr": trocr_result,
                "nlp": nlp_result,
                "grade": (grade_letter, grade_desc),
                "error_rates": error_rates,
                "cross_agreement": cross_agreement,
            }

        finally:
            # Nettoyage
            try:
                os.unlink(temp_path)
                if enable_preprocess and prep_path and os.path.exists(prep_path):
                    os.unlink(prep_path)
            except OSError:
                pass

    # ── Afficher les résultats ──
    if "results" in st.session_state:
        results = st.session_state["results"]
        easyocr_result = results["easyocr"]
        paddleocr_result = results.get("paddleocr")
        trocr_result = results["trocr"]
        nlp_result = results["nlp"]
        grade_letter, grade_desc = results["grade"]
        error_rates = results["error_rates"]
        cross_agreement = results["cross_agreement"]

        # ════════════════════════════════════════════════════════════
        # SECTION 1: Texte extrait
        # ════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section-header">📝 Texte extrait</div>
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

        if paddleocr_result and paddleocr_result.get("text"):
            st.markdown(f"""
            <div class="extracted-text" style="border-left-color: #FF6B9D; margin-top: 12px;">
                <span class="engine-tag" style="background: rgba(255,107,157,0.12); color: #FF6B9D; border-color: rgba(255,107,157,0.25);">🚀 PaddleOCR</span>
                <p style="margin-top: 12px; margin-bottom: 0;">{paddleocr_result['text']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        if trocr_result and trocr_result.get("text"):
            st.markdown(f"""
            <div class="extracted-text" style="border-left-color: #00D4AA; margin-top: 12px;">
                <span class="engine-tag" style="background: rgba(0,212,170,0.12); color: #00D4AA; border-color: rgba(0,212,170,0.25);">🌐 TrOCR</span>
                <p style="margin-top: 12px; margin-bottom: 0;">{trocr_result['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif enable_trocr and not trocr_result:
            st.info("ℹ️ TrOCR n'a pas pu se connecter. Vérifiez votre token HF.")

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
                "Confiance moyenne", conf_color
            ), unsafe_allow_html=True)

        with col_m2:
            min_color = get_confidence_color(easyocr_result["min_confidence"])
            st.markdown(render_metric_card(
                "📉", f"{easyocr_result['min_confidence']:.0%}",
                "Confiance min", min_color
            ), unsafe_allow_html=True)

        with col_m3:
            dict_color = get_confidence_color(nlp_result["dictionary_hit_rate"])
            st.markdown(render_metric_card(
                "📖", f"{nlp_result['dictionary_hit_rate']:.0%}",
                "Taux dictionnaire", dict_color
            ), unsafe_allow_html=True)

        with col_m4:
            st.markdown(render_metric_card(
                "🔢", str(nlp_result["word_count"]),
                "Mots détectés", "#8B83FF"
            ), unsafe_allow_html=True)

        # ── Ligne 2: Temps + CER/WER ──
        extra_cols = []
        if error_rates:
            extra_cols.append(("CER", error_rates["cer_percent"], "Character Error Rate"))
            extra_cols.append(("WER", error_rates["wer_percent"], "Word Error Rate"))
        if cross_agreement:
            extra_cols.append(("🤝", cross_agreement["similarity_percent"], "Concordance moteurs"))

        extra_cols.append(("⏱️", f"{easyocr_result['elapsed_seconds']}s", "Temps EasyOCR"))

        if paddleocr_result:
            extra_cols.append(("🚀", f"{paddleocr_result['elapsed_seconds']}s", "Temps PaddleOCR"))

        if trocr_result:
            extra_cols.append(("🌐", f"{trocr_result['elapsed_seconds']}s", "Temps TrOCR"))

        if extra_cols:
            cols = st.columns(len(extra_cols))
            for i, (icon, val, label) in enumerate(extra_cols):
                with cols[i]:
                    st.markdown(render_metric_card(icon, val, label), unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════
        # SECTION 3: Concordance inter-moteurs
        # ════════════════════════════════════════════════════════════
        if cross_agreement:
            st.markdown("""
            <div class="section-header">🤝 Concordance inter-moteurs</div>
            """, unsafe_allow_html=True)

            sim = cross_agreement["similarity"]
            bar_color = get_confidence_color(sim)

            col_bar, col_status = st.columns([2, 1])
            with col_bar:
                st.progress(sim, text=f"Similarité: {cross_agreement['similarity_percent']}")
            with col_status:
                st.markdown(cross_agreement["status"])

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
                <div style="font-size: 2rem;">🔍</div>
                <p style="color: #666; font-size: 0.9rem;">Double moteur OCR<br><small>EasyOCR + TrOCR</small></p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">📊</div>
                <p style="color: #666; font-size: 0.9rem;">Métriques de qualité<br><small>Confiance, CER, WER</small></p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🔤</div>
                <p style="color: #666; font-size: 0.9rem;">Validation NLP<br><small>Dictionnaire, corrections</small></p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem;">🤝</div>
                <p style="color: #666; font-size: 0.9rem;">Cross-validation<br><small>Concordance moteurs</small></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
