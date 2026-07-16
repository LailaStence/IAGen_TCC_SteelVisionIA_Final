# =============================================================================
# SteelVision AI — Ferramenta: Estatísticas do Dataset
#
# Tool disponibilizada ao LLM para consultar estatísticas sobre o dataset
# de defeitos em chapas de aço e dados de predição.
#
# Justificativa: O LLM precisa de dados quantitativos reais para contextualizar
# seus relatórios e não gerar estatísticas inventadas.
# =============================================================================

import json
import os

# Estatísticas do dataset NEU Surface Defect Database
# Referência: Song & Yan (2013) — Applied Surface Science, 285, 858-864

DATASET_STATS = {
    "total_images": 1800,
    "image_size": "200x200 pixels",
    "classes": 6,
    "per_class": 300,
    "resolution": "500x500 pixels (original), 200x200 pixels (treinamento)",
    "split": {
        "train": "80%",
        "validation": "20%"
    },
    "class_details": {
        "Crazing": {
            "count": 300,
            "description": "Fissuras finas e interconectadas"
        },
        "Inclusion": {
            "count": 300,
            "description": "Partículas estranhas incorporadas"
        },
        "Patches": {
            "count": 300,
            "description": "Áreas irregulares de coloração"
        },
        "Pitted Surface": {
            "count": 300,
            "description": "Cavidades ou pequenas crateras"
        },
        "Rolled-in Scale": {
            "count": 300,
            "description": "Óxido laminado incorporado"
        },
        "Scratches": {
            "count": 300,
            "description": "Arranhões lineares na superfície"
        }
    }
}


def get_dataset_info() -> dict:
    """
    Retorna informações gerais sobre o dataset.

    Returns:
        Dicionário com estatísticas do dataset.
    """
    return DATASET_STATS


def get_class_distribution() -> list:
    """
    Retorna a distribuição de imagens por classe.

    Returns:
        Lista de dicionários com nome da classe e quantidade.
    """
    return [
        {
            "class": cls,
            "count": info["count"],
            "description": info["description"]
        }
        for cls, info in DATASET_STATS["class_details"].items()
    ]


def get_prediction_history() -> list:
    """
    Retorna o histórico de predições registradas.

    Returns:
        Lista de predições do histórico JSON.
    """
    historico_path = "historico_predicoes.json"
    if os.path.exists(historico_path):
        with open(historico_path, "r") as f:
            return json.load(f)
    return []


def get_prediction_statistics() -> dict:
    """
    Calcula estatísticas do histórico de predições.

    Returns:
        Dicionário com distribuição, total e média de confiança.
    """
    historico = get_prediction_history()

    if not historico:
        return {
            "total_predictions": 0,
            "distribution": {},
            "average_confidence": 0
        }

    distribution = {}
    total_confidence = 0

    for pred in historico:
        classe = pred["classe"]
        distribution[classe] = distribution.get(classe, 0) + 1
        total_confidence += pred["confianca"]

    return {
        "total_predictions": len(historico),
        "distribution": distribution,
        "average_confidence": round(total_confidence / len(historico), 2)
    }
