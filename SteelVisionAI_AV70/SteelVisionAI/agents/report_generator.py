# =============================================================================
# SteelVision AI — Agente: Gerador de Relatórios via LLM
#
# Agente que utiliza um modelo LLM (via API) para gerar relatórios técnicos
# detalhados sobre defeitos detectados. Usa tool calling para acessar
# informações estruturadas sobre defeitos.
#
# Framework: Chamadas diretas à API OpenAI-compatible
# Modelo: Configurável (GPT-4o-mini por padrão)
# Estratégia de Prompting: Few-shot + Chain-of-thought
# =============================================================================

import json
import os
from datetime import datetime
import google.generativeai as genai
import streamlit as st
import os



class ReportGenerator:
    """
    Agente gerador de relatórios que utiliza LLM para produzir
    análises técnicas detalhadas sobre defeitos em chapas de aço.

    CONFIGURAÇÃO DE PARÂMETROS (documentada para o AV_70):
    - temperatura: 0.7 (equilíbrio entre criatividade e precisão técnica)
    - top_p: 0.9 (nucleus sampling para diversidade controlada)
    - max_tokens: 500 (relatórios concisos e focados)
    - presence_penalty: 0.0 (não penaliza repetição de termos técnicos)
    - frequency_penalty: 0.0 (não penaliza termos de domínio)

    JUSTIFICATIVAS:
    - Temperatura 0.7: Relatórios técnicos precisam de precisão, mas não
      devem ser robóticos. 0.7 permite variação natural sem inventar dados.
    - Top-p 0.9: Restringe o espaço de tokens para opções prováveis,
      reduzindo alucinações sem eliminar diversidade.
    - Max tokens 500: Relatórios devem ser concisos e objetivos.
    """

    DEFAULT_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 500,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }

    def __init__(self, config=None):
        """
        Inicializa o gerador de relatórios.

        Args:
            config: Dicionário com configurações do LLM.
                    Se None, usa DEFAULT_CONFIG.
        """
        self.config = config or self.DEFAULT_CONFIG.copy()
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        self.system_prompt = self._load_system_prompt()
        self.few_shot_examples = self._load_few_shot()
        self.model = genai.GenerativeModel( model_name="gemini-1.5-pro",
        system_instruction=self.system_prompt
        )

    def _load_system_prompt(self) -> str:
        """Carrega o system prompt do arquivo."""
        path = os.path.join("prompts", "system_prompt.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "Você é um assistente de engenharia de qualidade para chapas de aço."

    def _load_few_shot(self) -> str:
        """Carrega os exemplos few-shot do arquivo."""
        path = os.path.join("prompts", "few_shot_examples.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _load_report_prompt(self) -> str:
        """Carrega o prompt de relatório."""
        path = os.path.join("prompts", "report_prompt.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read().strip()
        return "Gere um relatório sobre o defeito {classe}."

    def generate_report(self, defect_class: str, confidence: float,
                        image_name: str = "") -> dict:
        """
        Gera um relatório técnico usando o LLM.

        Se a API não estiver disponível, usa o fallback local.

        Args:
            defect_class: Classe do defeito detectado.
            confidence: Nível de confiança da CNN.
            image_name: Nome da imagem analisada.

        Returns:
            Dicionário com o relatório e metadados.
        """
        # Tenta usar LLM
        llm_report = self._try_llm_report(defect_class, confidence, image_name)

        # Fallback para método local
        if llm_report is None:
            from agents.quality_assistant import QualityAssistant
            agent = QualityAssistant()
            analysis = agent.analyze(defect_class, confidence, image_name)
            text_report = agent.generate_text_report(analysis)
            return {
                "report": text_report,
                "source": "local_fallback",
                "model_used": "QualityAssistant (sem LLM)",
                "params": {
                    "temperature": 0,
                    "top_p": 1.0,
                    "max_tokens": "N/A"
                }
            }

        return {
            "report": llm_report,
            "source": "llm",
            "model_used": self.config["model"],
            "params": {
                "temperature": self.config["temperature"],
                "top_p": self.config["top_p"],
                "max_tokens": self.config["max_tokens"]
            }
        }

    def _try_llm_report(self, defect_class: str, confidence: float,
                        image_name: str) -> str | None:
        """
        Tenta gerar relatório via LLM.

        Returns:
            String com o relatório ou None se falhar.
        """
        if not self.api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.api_base)

            # Montar mensagem do usuário com contexto
            user_message = (
                f"Análise de defeito detectado:\n"
                f"- Classe: {defect_class}\n"
                f"- Confiança: {confidence}%\n"
                f"- Arquivo: {image_name}\n\n"
                f"Gere o relatório técnico seguindo a estrutura definida no prompt."
            )

            # Montar mensagens com few-shot
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]

            # Adicionar few-shot examples
            if self.few_shot_examples:
                messages.append({"role": "assistant", "content": self.few_shot_examples})

            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                max_tokens=self.config["max_tokens"]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Erro ao chamar LLM: {e}")
            return None

    def update_config(self, **kwargs):
        """
        Atualiza parâmetros de configuração do LLM.

        Args:
            **kwargs: Parâmetros para atualizar (temperature, top_p, model, etc.)
        """
        self.config.update(kwargs)
