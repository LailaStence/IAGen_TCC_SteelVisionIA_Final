# =============================================================================
# SteelVision AI — Agente: Assistente de Qualidade
#
# Agente principal que coordena o fluxo de análise de qualidade.
# Recebe a classificação da CNN e utiliza ferramentas do LLM para gerar
# uma análise completa e contextualizada do defeito.
#
# Arquitetura: Agente single-turn com tool calling
# Framework: Chamadas diretas à API (sem LangChain)
# =============================================================================

import json
import os
from datetime import datetime

from tools.defect_database import consult_defect, list_defects, search_defects_by_severity
from tools.correction_guide import get_corrections, list_corrections_summary
from tools.statistics_tool import get_dataset_info, get_prediction_statistics


class QualityAssistant:
    """
    Agente assistente de qualidade que coordena a análise de defeitos.

    Este agente recebe a classificação da CNN e orquestra as ferramentas
    disponíveis para construir uma análise completa. Ele NÃO usa LLM —
    o agente constrói a análise diretamente a partir das tools, demonstrando
    que o fluxo funciona mesmo sem dependência externa do LLM.

    Quando o LLM estiver configurado, este agente será a ponte entre
    a classificação CNN e o LLM, fornecendo dados estruturados.
    """

    def __init__(self):
        self.defect_db = None
        self.corrections = None
        self.dataset_info = None
        self.prediction_stats = None

    def analyze(self, defect_class: str, confidence: float, image_name: str = "") -> dict:
        """
        Realiza a análise completa de um defeito detectado.

        Args:
            defect_class: Classe do defeito detectado pela CNN.
            confidence: Nível de confiança da classificação (0-100).
            image_name: Nome do arquivo de imagem analisado.

        Returns:
            Dicionário com análise completa do defeito.
        """
        self.defect_db = consult_defect(defect_class)
        self.corrections = get_corrections(defect_class)
        self.dataset_info = get_dataset_info()
        self.prediction_stats = get_prediction_statistics()

        analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": image_name,
            "classification": {
                "defect_class": defect_class,
                "confidence": round(confidence, 2),
                "needs_review": confidence < 70
            },
            "defect_info": {
                "name": self.defect_db["name"],
                "description": self.defect_db["description"],
                "causes": self.defect_db["causes"],
                "severity": self.defect_db["severity"],
                "severity_reason": self.defect_db["severity_reason"],
                "detection_difficulty": self.defect_db["detection_difficulty"]
            },
            "corrections": {
                "immediate_actions": self.corrections["immediate_actions"],
                "preventive_measures": self.corrections["preventive_measures"],
                "process_adjustments": self.corrections["process_adjustments"],
                "cost_impact": self.corrections["estimated_cost_impact"]
            },
            "context": {
                "dataset_total_images": self.dataset_info["total_images"],
                "dataset_classes": self.dataset_info["classes"],
                "recent_predictions": self.prediction_stats["total_predictions"],
                "average_confidence": self.prediction_stats["average_confidence"]
            }
        }

        return analysis

    def generate_text_report(self, analysis: dict) -> str:
        """
        Gera um relatório em texto formatado para exibição.
        Este método constrói o relatório manualmente (sem LLM),
        servindo como fallback quando o LLM não está disponível.

        Args:
            analysis: Dicionário de análise gerado pelo método analyze().

        Returns:
            String com o relatório formatado.
        """
        lines = []
        lines.append(f"## Relatório de Qualidade — SteelVision AI")
        lines.append("")
        lines.append(f"**Defeito Detectado:** {analysis['classification']['defect_class']}")
        lines.append(f"**Confiança da Classificação:** {analysis['classification']['confidence']}%")
        lines.append(f"**Data da Análise:** {analysis['timestamp']}")

        if analysis['classification']['needs_review']:
            lines.append("")
            lines.append("**⚠️ AVISO:** Confiança abaixo de 70%. Verificação manual recomendada.")

        lines.append("")
        lines.append("### 1. Descrição do Defeito")
        lines.append(analysis['defect_info']['description'])

        lines.append("")
        lines.append("### 2. Causa Provável")
        for cause in analysis['defect_info']['causes']:
            lines.append(f"- {cause}")

        lines.append("")
        lines.append("### 3. Severidade Estimada")
        lines.append(f"**{analysis['defect_info']['severity']}** — {analysis['defect_info']['severity_reason']}")

        lines.append("")
        lines.append("### 4. Ações Recomendadas")
        for action in analysis['corrections']['immediate_actions']:
            lines.append(f"1. {action}")

        lines.append("")
        lines.append("### 5. Prevenção")
        for measure in analysis['corrections']['preventive_measures']:
            lines.append(f"- {measure}")

        lines.append("")
        lines.append(f"**Impacto de Custo Estimado:** {analysis['corrections']['cost_impact']}")

        return "\n".join(lines)

    def get_tool_definitions(self) -> list:
        """
        Retorna as definições de ferramentas no formato OpenAI.
        Usado para configurar tool calling do LLM.

        Returns:
            Lista de definições de ferramentas no formato OpenAI.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "consult_defect",
                    "description": "Consulta informações técnicas sobre um tipo de defeito em chapas de aço",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "defect_class": {
                                "type": "string",
                                "description": "Nome da classe do defeito",
                                "enum": ["Crazing", "Inclusion", "Patches", "Pitted Surface", "Rolled-in Scale", "Scratches"]
                            }
                        },
                        "required": ["defect_class"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_corrections",
                    "description": "Consulta ações corretivas e preventivas para um defeito específico",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "defect_class": {
                                "type": "string",
                                "description": "Nome da classe do defeito"
                            }
                        },
                        "required": ["defect_class"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_prediction_statistics",
                    "description": "Obtém estatísticas das predições recentes do sistema",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
