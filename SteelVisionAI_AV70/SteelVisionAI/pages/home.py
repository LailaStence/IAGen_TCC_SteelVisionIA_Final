import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠")

st.title("🏠 SteelVision AI")

st.markdown("""
## Sistema Inteligente para Detecção de Defeitos em Chapas de Aço

Este sistema foi desenvolvido como complemento do Trabalho de Conclusão de Curso (TCC),
utilizando técnicas de Visão Computacional e Inteligência Artificial para identificar
automaticamente defeitos superficiais em chapas de aço.

### Objetivos

- Automatizar a inspeção visual.
- Reduzir falhas humanas.
- Auxiliar no controle de qualidade industrial.
- Demonstrar o uso de Redes Neurais Convolucionais (CNN).

---

### Tecnologias

- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- Scikit-Learn
- Kaggle NEU-DET Dataset

""")

col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Classes",6)

with col2:
    st.metric("Modelo","CNN")

with col3:
    st.metric("Framework","Streamlit")

st.divider()

st.info("Utilize o menu lateral para navegar pelas páginas do sistema.")
