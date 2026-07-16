# =============================================================================
# SteelVision AI — Treinamento do Modelo
#
# Interface para iniciar o treinamento da CNN com controle de hiperparâmetros.
# Exibe progresso em tempo real e resultados finais.
# =============================================================================

import streamlit as st
from utils.trainer import treinar_modelo

st.set_page_config(page_title="Treinamento", page_icon="🧠")

st.title("🧠 Treinamento da CNN")

st.markdown("""
<div class="card">
    <p>Nesta etapa, o modelo de Rede Neural Convolucional (CNN) será treinado utilizando o dataset
    <strong>NEU Surface Defect Database</strong>. O modelo utiliza 3 camadas convolucionais
    (32, 64, 128 filtros) seguidas de camadas densas para classificação em 6 categorias de defeitos.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# PARÂMETROS
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    epocas = st.slider(
        "Número de Épocas",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        key="slider_epocas"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    batch = st.selectbox(
        "Batch Size",
        options=[8, 16, 32],
        index=2,
        key="select_batch"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Informações sobre o modelo
st.markdown("#### Arquitetura do Modelo")
st.markdown("""
| Camada | Tipo | Detalhes |
| :--- | :--- | :--- |
| 1 | Rescaling | Normalização 0-1 |
| 2 | Conv2D | 32 filtros, kernel 3x3, ReLU |
| 3 | MaxPooling2D | Pooling 2x2 |
| 4 | Conv2D | 64 filtros, kernel 3x3, ReLU |
| 5 | MaxPooling2D | Pooling 2x2 |
| 6 | Conv2D | 128 filtros, kernel 3x3, ReLU |
| 7 | MaxPooling2D | Pooling 2x2 |
| 8 | Flatten | Vetorização |
| 9 | Dense | 256 unidades, ReLU |
| 10 | Dropout | 40% de dropout |
| 11 | Dense | 6 unidades, Softmax (saída) |
""")

# =============================================================================
# TREINAMENTO
# =============================================================================

if st.button("🚀 Treinar Modelo", use_container_width=True, key="btn_treinar"):

    st.divider()
    st.markdown("### Treinamento em Andamento...")

    progresso = st.progress(0)
    status = st.empty()

    def atualizar(valor):
        progresso.progress(valor)

    try:
        historico = treinar_modelo(
            epochs=epocas,
            batch_size=batch,
            callback=atualizar,
            status=status
        )

        st.divider()
        st.success("✅ Treinamento concluído com sucesso!")

        # Exibir métricas finais
        st.subheader("Resultados")
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
            st.metric("Acurácia (Treino)", f"{historico['accuracy'][-1]:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
            st.metric("Acurácia (Val.)", f"{historico['val_accuracy'][-1]:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
            st.metric("Loss (Treino)", f"{historico['loss'][-1]:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
            st.metric("Loss (Val.)", f"{historico['val_loss'][-1]:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Gráficos
        st.subheader("Curvas de Treinamento")
        col_g1, col_g2 = st.columns(2)

        import matplotlib.pyplot as plt
        import numpy as np

        with col_g1:
            fig, ax = plt.subplots(figsize=(6, 4))
            eps = range(1, len(historico["accuracy"]) + 1)
            ax.plot(eps, historico["accuracy"], "b-o", label="Treino", markersize=4)
            ax.plot(eps, historico["val_accuracy"], "r-s", label="Validação", markersize=4)
            ax.set_title("Acurácia")
            ax.set_xlabel("Época")
            ax.set_ylabel("Acurácia")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col_g2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            eps = range(1, len(historico["loss"]) + 1)
            ax2.plot(eps, historico["loss"], "b-o", label="Treino", markersize=4)
            ax2.plot(eps, historico["val_loss"], "r-s", label="Validação", markersize=4)
            ax2.set_title("Loss")
            ax2.set_xlabel("Época")
            ax2.set_ylabel("Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        st.info("O modelo foi salvo em **models/modelo.keras** e está pronto para classificação.")

    except Exception as e:
        st.error(f"❌ Erro durante o treinamento: {str(e)}")
        st.exception(e)
