import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OCR Insights",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    /* Global styles copied from app.py to maintain consistency */
    :root {
        --bg-main: #0B0E14;
        --bg-card: #151822;
        --primary: #6C63FF;
        --success: #00D4AA;
        --danger: #FF5C5C;
        --text-primary: #FFFFFF;
        --text-secondary: #A0A4B8;
        --border: #2A2D3D;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-color: var(--primary);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 12px 0 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 OCR Insights & Analytics")
st.markdown("Analysez les performances et la fiabilité de vos extractions OCR dans le temps.")

HISTORY_FILE = "ocr_history.json"

if not os.path.exists(HISTORY_FILE):
    st.info("Aucune donnée d'historique trouvée. Veuillez d'abord scanner une image dans l'application principale.")
else:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not data:
            st.info("L'historique est vide.")
        else:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Top metrics
            total_scans = len(df)
            avg_trust = df['trust_score'].mean() * 100
            avg_conf = df['avg_confidence'].mean() * 100
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">📝</div>
                    <div class="metric-value" style="color: #6C63FF;">{total_scans}</div>
                    <div class="metric-label">Images Scannées</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">🤖</div>
                    <div class="metric-value" style="color: #00D4AA;">{avg_trust:.1f}%</div>
                    <div class="metric-label">Confiance IA Moyenne</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">🎯</div>
                    <div class="metric-value" style="color: #5BA4FF;">{avg_conf:.1f}%</div>
                    <div class="metric-label">Confiance OCR Moyenne</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                # Count grades
                best_grade = df['grade'].value_counts().idxmax()
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">🏆</div>
                    <div class="metric-value" style="color: #FFB84D;">{best_grade}</div>
                    <div class="metric-label">Grade le plus fréquent</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Évolution de la confiance")
                chart_data = df.set_index('timestamp')[['trust_score', 'avg_confidence']]
                chart_data.columns = ['AI Trust Score', 'EasyOCR Confidence']
                st.line_chart(chart_data)
                
            with col_chart2:
                st.subheader("Répartition des grades")
                grade_counts = df['grade'].value_counts()
                st.bar_chart(grade_counts)
            
            st.markdown("---")
            st.subheader("📋 Historique complet")
            
            # Format the dataframe for display
            display_df = df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['trust_score'] = (display_df['trust_score'] * 100).map('{:.1f}%'.format)
            display_df['avg_confidence'] = (display_df['avg_confidence'] * 100).map('{:.1f}%'.format)
            display_df['engines_used'] = display_df['engines_used'].apply(lambda x: ", ".join(x))
            
            display_df = display_df[['timestamp', 'filename', 'grade', 'trust_score', 'avg_confidence', 'engines_used', 'text']]
            display_df.columns = ['Date', 'Fichier', 'Grade', 'Trust AI', 'Conf OCR', 'Moteurs', 'Texte (Aperçu)']
            display_df['Texte (Aperçu)'] = display_df['Texte (Aperçu)'].str.slice(0, 50) + "..."
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"Erreur lors de la lecture de l'historique : {e}")
