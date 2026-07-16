# =============================================================================
# SteelVision AI v2.0 — Sistema de Visão Computacional + IA Generativa
#
# Sistema integrado que combina:
#   1. CNN para classificação visual de defeitos em chapas de aço
#   2. LLM (IA Generativa) para geração de relatórios técnicos
#
# Desenvolvedora: Laila Michely Stence
# Pós-Graduação em Inteligência Artificial Aplicada
#
# Arquitetura LLM:
#   - Framework: Chamadas diretas à API OpenAI-compatible
#   - Modelo: GPT-4o-mini (configurável)
#   - System Prompt: prompts/system_prompt.txt
#   - Tools: tools/defect_database.py, tools/correction_guide.py, tools/statistics_tool.py
#   - Agents: agents/quality_assistant.py, agents/report_generator.py
# =============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import google.generativeai as genai



# =============================================================================
# Configuração da página
# =============================================================================

st.set_page_config(
    page_title="SteelVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar CSS personalizado
if os.path.exists("style.css"):
    with open("style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

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
    "Crazing": "Fissuras finas e interconectadas na superfície do aço, geralmente causadas por tensões residuais durante o resfriamento.",
    "Inclusion": "Partículas estranhas ou inclusões não metálicas incorporadas à superfície durante o processo de laminação.",
    "Patches": "Áreas irregulares de coloração diferente na superfície, resultantes de oxidação localizada ou contaminação.",
    "Pitted Surface": "Cavidades ou pequenas crateras na superfície, normalmente causadas por corrosão ou erosão.",
    "Rolled-in Scale": "Óxido laminado incorporado à superfície durante o processo de laminação a quente.",
    "Scratches": "Arranhões lineares na superfície, causados por contato com equipamentos ou materiais durante o manuseio."
}

IMG_SIZE = 200
MODEL_PATH = "models/modelo.keras"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def preprocessar(imagem):
    """Preprocessa uma imagem PIL para entrada do modelo CNN."""
    imagem = imagem.convert("RGB")
    imagem = imagem.resize((IMG_SIZE, IMG_SIZE))
    imagem = np.array(imagem).astype("float32") / 255.0
    imagem = np.expand_dims(imagem, axis=0)
    return imagem


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo Keras treinado (CNN)."""
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


def carregar_historico():
    """Carrega o histórico de predições."""
    historico_path = "historico_predicoes.json"
    if os.path.exists(historico_path):
        with open(historico_path, "r") as f:
            return json.load(f)
    return []


def get_llm_report(agent, defect_class, confidence, image_name=""):
    """
    Gera relatório usando o agente LLM (ReportGenerator).
    Faz fallback automático para o agente local se o LLM não estiver disponível.
    """
    return agent.generate_report(defect_class, confidence, image_name)


# =============================================================================
# CARREGAR MODELO CNN
# =============================================================================

modelo_ok = False
try:
    modelo = carregar_modelo()
    modelo_ok = True
except Exception as e:
    modelo_ok = False

# =============================================================================
# INICIALIZAR AGENTES
# =============================================================================

from agents.quality_assistant import QualityAssistant
from agents.report_generator import ReportGenerator

quality_agent = QualityAssistant()

# Configurar gerador de relatórios
report_config = st.session_state.get("llm_config", {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 500
})
report_agent = ReportGenerator(config=report_config)

# =============================================================================
# NAVEGAÇÃO
# =============================================================================

pg = st.navigation([
    st.Page(lambda: None, title="Início", icon="🏠"),
    st.Page("pages/predicao.py", title="Predição", icon="🤖"),
    st.Page("pages/dataset.py", title="Dataset", icon="📂"),
    st.Page("pages/analise.py", title="Análise", icon="📊"),
    st.Page("pages/treinamento.py", title="Treinamento", icon="🧠"),
    st.Page("pages/sobre.py", title="Sobre", icon="ℹ️"),
])

# =============================================================================
# PÁGINA PRINCIPAL — INÍCIO
# =============================================================================
if pg.title == "Início":

    # Cabeçalho
    st.title("🔍 SteelVision AI v2.0")

    st.markdown("""
    <div class="card" style="padding: 25px; margin-bottom: 25px;">
        <h3 style="color: #0A4D8C; margin-top: 0;">Sistema Inteligente com IA Generativa para Detecção de Defeitos em Chapas de Aço</h3>
        <p>Este sistema integra duas IAs: uma <strong>Rede Neural Convolucional (CNN)</strong> para classificação visual 
        de defeitos e um <strong>Modelo de Linguagem Grande (LLM)</strong> para geração de relatórios técnicos. 
        A IA generativa utiliza tool calling para consultar bases de conhecimento estruturadas e produzir 
        análises precisas e contextualizadas.</p>
    </div>
    """, unsafe_allow_html=True)

    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    # Verificar LLM
    llm_available = bool(st.secrets.get("GOOGLE_API_KEY", ""))

    if not modelo_ok:
        st.warning("⚠️ Modelo CNN não encontrado. Treine o modelo na aba 'Treinamento'.")

    if not llm_available:
        st.info("ℹ️ API do LLM não configurada. Os relatórios serão gerados localmente. Configure a chave API para usar IA Generativa.")

    # =============================================================================
    # SEÇÃO 1 — UPLOAD E CLASSIFICAÇÃO COM LLM
    # =============================================================================

    st.subheader("📷 Classificação + Relatório com IA Generativa")

    col_upload, col_resultado = st.columns([1, 1])

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Faça o Upload da Imagem")
        st.caption("Formatos: JPG, PNG, JPEG, BMP")

        arquivo = st.file_uploader(
            "Escolha uma imagem de chapa de aço",
            type=["jpg", "png", "jpeg", "bmp"],
            key=f"upload_principal_{time.time()}"
        )

        if arquivo is not None:
            imagem = Image.open(arquivo)
            st.image(imagem, width=300, caption="Imagem Carregada")

            if modelo_ok:
                if st.button("🔍 Classificar e Gerar Relatório", use_container_width=True, key="btn_classificar"):
                    # Processar imagem com CNN
                    img_processada = preprocessar(imagem)
                    pred = modelo.predict(img_processada, verbose=0)
                    indice = np.argmax(pred)
                    confianca = float(np.max(pred)) * 100

                    todas_probs = {c: round(p * 100, 2) for c, p in zip(CLASSES, pred[0].tolist())}
                    defeito = CLASSES[indice]

                    # Análise com agente de qualidade
                    analysis = quality_agent.analyze(defeito, confianca, arquivo.name)

                    # Relatório via LLM
                    report = get_llm_report(report_agent, defeito, confianca, arquivo.name)

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    resultado = {
                        "data": timestamp,
                        "classe": defeito,
                        "confianca": round(confianca, 2),
                        "probabilidades": todas_probs,
                        "report_source": report["source"],
                        "report_model": report["model_used"]
                    }
                    salvar_historico(resultado)

                    st.session_state["ultima_predicao"] = resultado
                    st.session_state["ultima_analise"] = analysis
                    st.session_state["ultimo_report"] = report
                    st.rerun()
            else:
                st.info("Treine o modelo na aba 'Treinamento' para ativar a classificação.")

            st.markdown('</div>', unsafe_allow_html=True)

    with col_resultado:
        if "ultima_predicao" in st.session_state:
            res = st.session_state["ultima_predicao"]
            analysis = st.session_state.get("ultima_analise")
            report = st.session_state.get("ultimo_report")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Resultado da Classificação")

            # Classe e confiança
            st.success(f"**Classe: {res['classe']}**")
            st.metric("Confiança", f"{res['confianca']:.2f}%")

            # Fonte do relatório
            fonte_badge = "✅ LLM" if res.get("report_source") == "llm" else "🔄 Local"
            st.caption(f"Relatório gerado por: **{res.get('report_model', 'N/A')}** {fonte_badge}")

            # Probabilidades
            st.markdown("#### Probabilidades por Classe")
            fig, ax = plt.subplots(figsize=(8, 4))
            classes_ord = sorted(res["probabilidades"].items(), key=lambda x: x[1], reverse=True)
            nomes = [c[0] for c in classes_ord]
            probs = [c[1] for c in classes_ord]
            cores = ['#F57C00' if i == 0 else '#0A4D8C' for i in range(len(nomes))]
            bars = ax.barh(nomes, probs, color=cores)
            ax.set_xlabel("Probabilidade (%)")
            ax.set_xlim(0, 100)
            ax.set_title("Distribuição de Probabilidade")
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f"{prob:.1f}%", va='center', fontsize=10)
            st.pyplot(fig)

            st.markdown('</div>', unsafe_allow_html=True)

            # Relatório LLM
            if report:
                st.divider()
                st.subheader("📄 Relatório da IA Generativa")

                st.markdown('<div class="card" style="border-left-color: #F57C00;">', unsafe_allow_html=True)

                if report["source"] == "llm":
                    st.caption(f"Gerado por: {report['model_used']} | "
                            f"Temperatura: {report['params']['temperature']} | "
                            f"Top-p: {report['params']['top_p']} | "
                            f"Max Tokens: {report['params']['max_tokens']}")
                else:
                    st.caption(f"Gerado localmente (LLM não disponível) | "
                            f"Agente: {report['model_used']}")

                st.markdown(report["report"])

                if analysis:
                    st.divider()
                    st.markdown("#### Análise Estruturada (Agente)")
                    st.markdown(f"**Severidade:** {analysis['defect_info']['severity']}")
                    st.markdown(f"**Causas:**")
                    for cause in analysis['defect_info']['causes']:
                        st.markdown(f"- {cause}")
                    st.markdown(f"**Ações Imediatas:**")
                    for action in analysis['corrections']['immediate_actions']:
                        st.markdown(f"- {action}")

                st.markdown('</div>', unsafe_allow_html=True)

        elif modelo_ok and arquivo is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("Clique no botão **'Classificar e Gerar Relatório'** para analisar com IA.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("Faça upload de uma imagem para iniciar a análise com IA.")
            st.markdown('</div>', unsafe_allow_html=True)

    # =============================================================================
    # SEÇÃO 2 — CONFIGURAÇÃO DO LLM
    # =============================================================================

    st.divider()
    st.subheader("⚙️ Configuração da IA Generativa (LLM)")

    st.markdown("""
    <div class="card">
        <p>Configure os parâmetros do modelo LLM utilizado para geração de relatórios técnicos.
        Cada parâmetro afeta diretamente a qualidade e o comportamento das respostas.</p>
    </div>
    """, unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        model = st.selectbox(
            "Modelo",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0,
            key=f"select_model_{time.time()}"
        )
        st.caption("Modelo de linguagem usado pelo LLM")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            format="%.1f",
            key="slider_temperature"
        )
        st.caption("0.0 = determinístico, 1.0 = criativo")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1,
            format="%.1f",
            key="slider_top_p"
        )
        st.caption("Restringe tokens aos mais prováveis")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        max_tokens = st.selectbox(
            "Max Tokens",
            options=[256, 500, 750, 1000],
            index=1,
            key="select_max_tokens"
        )
        st.caption("Limite de tokens na resposta")
        st.markdown('</div>', unsafe_allow_html=True)

    # Botão para aplicar configuração
    if st.button("💾 Aplicar Configuração", use_container_width=True, key="btn_config_llm"):
        st.session_state["llm_config"] = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        report_agent.update_config(**st.session_state["llm_config"])
        st.success("Configuração aplicada com sucesso!")
        st.rerun()

    # =============================================================================
    # SEÇÃO 3 — FERRAMENTAS DISPONÍVEIS AO LLM
    # =============================================================================

    st.divider()
    st.subheader("🛠️ Ferramentas (Tools) Disponíveis ao LLM")

    st.markdown("""
    <div class="card">
        <p>O LLM tem acesso às seguintes ferramentas tipadas para consulta de informações estruturadas.
        Estas ferramentas evitam que o modelo invente dados sobre defeitos.</p>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar definições de tools
    tool_defs = quality_agent.get_tool_definitions()
    for tool in tool_defs:
        fn = tool["function"]
        with st.expander(f"🔧 `{fn['name']}` — {fn['description']}"):
            st.markdown(f"**Parâmetros:** {json.dumps(fn['parameters'], indent=2, ensure_ascii=False)}")

    # =============================================================================
    # SEÇÃO 4 — AGENTES DO SISTEMA
    # =============================================================================

    st.divider()
    st.subheader("🤖 Agentes do Sistema")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Quality Assistant")
        st.markdown("Agente principal que coordena a análise de defeitos, "
                    "consultando as ferramentas de base de dados e guia de correções. "
                    "Funciona como fallback quando o LLM não está disponível.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_a2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Report Generator")
        st.markdown("Agente que utiliza o LLM para gerar relatórios técnicos detalhados. "
                    "Usa tool calling para acessar informações estruturadas e produz "
                    "relatórios com Chain-of-thought.")
        st.markdown('</div>', unsafe_allow_html=True)

    # =============================================================================
    # SEÇÃO 5 — HISTÓRICO DE PREDIÇÕES
    # =============================================================================

    st.divider()
    st.subheader("📜 Histórico de Predições")

    historico = carregar_historico()

    if historico:
        dados_hist = []
        for p in historico[-10:]:
            dados_hist.append({
                "Data": p["data"],
                "Classe": p["classe"],
                "Confiança": f"{p['confianca']:.2f}%",
                "Fonte": p.get("report_source", "N/A")
            })
        df_hist = pd.DataFrame(dados_hist)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if st.button("🗑️ Limpar Histórico", key="btn_limpar_hist"):
            with open("historico_predicoes.json", "w") as f:
                json.dump([], f)
            st.rerun()
    else:
        st.info("Nenhuma predição registrada. Faça upload de uma imagem para começar.")

    # =============================================================================
    # SEÇÃO 6 — MÉTRICAS DO SISTEMA
    # =============================================================================

    st.divider()
    st.subheader("📊 Métricas do Sistema")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.metric("Classes", "6")
        st.caption("CNN Classification")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.metric("Tools", "3")
        st.caption("LLM Tool Calling")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m3:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.metric("Agents", "2")
        st.caption("Quality + Report")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m4:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.metric("Prompts", "4")
        st.caption("System + Report + Suggest + Few-shot")
        st.markdown('</div>', unsafe_allow_html=True)

    # =============================================================================
    # RUN NAVIGATION
# =============================================================================

pg.run()
