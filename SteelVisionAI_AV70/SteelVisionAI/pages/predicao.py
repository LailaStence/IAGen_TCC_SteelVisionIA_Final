# =============================================================================
# SteelVision AI — Página de Predição
#
# Interface dedicada para classificação de imagens de chapas de aço.
# Suporta upload múltiplo de imagens e exibe probabilidades por classe.
# =============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

st.set_page_config(page_title="Predição", page_icon="🤖")

st.title("🤖 Predição de Defeitos")

st.markdown("""
<div class="card">
    <p>Faça o upload de uma ou mais imagens de chapas de aço para identificar automaticamente
    o tipo de defeito superficial presente. O modelo CNN foi treinado com o dataset
    NEU Surface Defect Database e é capaz de classificar <strong>6 tipos de defeitos</strong>.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CLASSES E CONSTANTES
# =============================================================================

CLASSES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches"
]

CLASS_DESCRIPTIONS = {
    "Crazing": "Fissuras finas e interconectadas na superfície do aço.",
    "Inclusion": "Partículas estranhas incorporadas à superfície.",
    "Patches": "Áreas irregulares de coloração diferente na superfície.",
    "Pitted Surface": "Cavidades ou pequenas crateras na superfície.",
    "Rolled-in Scale": "Óxido laminado incorporado à superfície.",
    "Scratches": "Arranhões lineares na superfície."
}

IMG_SIZE = 200
MODEL_PATH = "models/modelo.keras"


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def preprocessar(imagem):
    """Preprocessa uma imagem PIL para entrada do modelo."""
    imagem = imagem.convert("RGB")
    imagem = imagem.resize((IMG_SIZE, IMG_SIZE))
    imagem = np.array(imagem).astype("float32") / 255.0
    imagem = np.expand_dims(imagem, axis=0)
    return imagem


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo Keras treinado."""
    return tf.keras.models.load_model(MODEL_PATH)


def salvar_historico(predicao):
    """Salva uma predição no histórico."""
    historico_path = "historico_predicoes.json"
    if os.path.exists(historico_path):
        with open(historico_path, "r") as f:
            historico = json.load(f)
    else:
        historico = []

    historico.append(predicao)
    if len(historico) > 50:
        historico = historico[-50:]

    with open(historico_path, "w") as f:
        json.dump(historico, f, indent=2)


# =============================================================================
# CARREGAR MODELO
# =============================================================================

modelo_ok = False
try:
    modelo = carregar_modelo()
    modelo_ok = True
except Exception as e:
    modelo_ok = False
    st.warning(f"⚠️ Modelo não encontrado em '{MODEL_PATH}'. Treine o modelo na aba 'Treinamento'.")

# =============================================================================
# UPLOAD E PREDIÇÃO
# =============================================================================

st.subheader("📷 Upload de Imagem")

arquivo = st.file_uploader(
    "Escolha uma imagem de chapa de aço",
    type=["jpg", "png", "jpeg", "bmp"],
    key="upload_predicao"
)

if arquivo is not None:
    imagem = Image.open(arquivo)
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(imagem, width=350, caption="Imagem Original")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if modelo_ok:
            if st.button("🔍 Classificar", use_container_width=True, key="btn_classificar_pag"):
                img_processada = preprocessar(imagem)
                pred = modelo.predict(img_processada, verbose=0)
                indice = np.argmax(pred)
                confianca = float(np.max(pred)) * 100

                # Salvar histórico
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                resultado = {
                    "data": timestamp,
                    "classe": CLASSES[indice],
                    "confianca": round(confianca, 2),
                    "probabilidades": {c: round(p * 100, 2) for c, p in zip(CLASSES, pred[0].tolist())}
                }
                salvar_historico(resultado)

                st.session_state["predicao_pagina"] = resultado
                st.rerun()
        else:
            st.info("Treine o modelo na aba 'Treinamento' para ativar a classificação.")

        st.markdown('</div>', unsafe_allow_html=True)

    # =============================================================================
    # RESULTADO
    # =============================================================================
    if "predicao_pagina" in st.session_state:
        res = st.session_state["predicao_pagina"]

        st.divider()
        st.subheader("Resultado da Classificação")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success(f"**Classe: {res['classe']}**")
            st.metric("Confiança", f"{res['confianca']:.2f}%")
            if res["classe"] in CLASS_DESCRIPTIONS:
                st.info(CLASS_DESCRIPTIONS[res["classe"]])
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Probabilidades")

            fig, ax = plt.subplots(figsize=(7, 4))
            classes_ord = sorted(res["probabilidades"].items(), key=lambda x: x[1], reverse=True)
            nomes = [c[0] for c in classes_ord]
            probs = [c[1] for c in classes_ord]
            cores = ['#F57C00' if i == 0 else '#0A4D8C' for i in range(len(nomes))]

            bars = ax.barh(nomes, probs, color=cores)
            ax.set_xlabel("Probabilidade (%)")
            ax.set_xlim(0, 100)
            ax.set_title("Distribuição de Probabilidade")

            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.1f}%", va='center', fontsize=10)

            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    elif modelo_ok and arquivo is not None:
        st.divider()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("Clique no botão **'Classificar'** para analisar a imagem.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.divider()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("Faça o upload de uma imagem para iniciar a classificação.")
        st.markdown('</div>', unsafe_allow_html=True)
