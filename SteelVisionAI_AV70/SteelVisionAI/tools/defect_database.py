# =============================================================================
# SteelVision AI — Ferramenta: Base de Dados de Defeitos
#
# Tool disponibilizada ao LLM para consultar informações técnicas sobre
# cada tipo de defeito superficial em chapas de aço.
#
# Justificativa: O LLM precisa de acesso a dados estruturados e confiáveis
# sobre defeitos para gerar relatórios precisos, sem inventar informações.
# =============================================================================

import json

# Base de conhecimento sobre defeitos em chapas de aço laminado
# Fonte: NEU Surface Defect Database — Song & Yan (2013)
# Referência: Applied Surface Science, 285, 858-864

DEFECT_DATABASE = {
    "Crazing": {
        "name": "Crazing (Fissuras)",
        "description": (
            "Fissuras finas e interconectadas na superfície do aço, formando um "
            "padrão reticulado característico. Geralmente causadas por tensões residuais "
            "durante o processo de resfriamento pós-laminação."
        ),
        "causes": [
            "Resfriamento não uniforme após laminação a quente",
            "Tensões térmicas residuais não aliviadas",
            "Variação na composição química do aço entre lotes"
        ],
        "severity": "Alta",
        "severity_reason": (
            "As fissuras podem propagar sob carga mecânica, comprometendo a "
            "resistência à fadiga do material em aplicações estruturais críticas."
        ),
        "detection_difficulty": "Moderada — requer análise de superfície em alta resolução"
    },
    "Inclusion": {
        "name": "Inclusion (Inclusões)",
        "description": (
            "Partículas estranhas ou inclusões não metálicas incorporadas à superfície "
            "do aço durante o processo de laminação. Podem ser óxidos, sulfetos ou "
            "silicatos."
        ),
        "causes": [
            "Contaminação durante o processo de fundição",
            "Desoxidantes não removidos adequadamente",
            "Falha na purificação do metal líquido"
        ],
        "severity": "Média",
        "severity_reason": (
            "As inclusões criam pontos de concentração de tensão que podem "
            "iniciar falhas sob carregamento, mas seu impacto depende do tamanho "
            "e da localização."
        ),
        "detection_difficulty": "Baixa — geralmente visível em inspeção visual"
    },
    "Patches": {
        "name": "Patches (Manchas)",
        "description": (
            "Áreas irregulares de coloração diferente na superfície, resultantes de "
            "oxidação localizada, contaminação química ou variações na composição "
            "superficial do aço."
        ),
        "causes": [
            "Oxidação localizada durante o processamento",
            "Contaminação por lubrificantes ou fluidos",
            "Variação na espessura da camada de óxido"
        ],
        "severity": "Baixa",
        "severity_reason": (
            "As manchas geralmente são superficiais e não comprometem as "
            "propriedades mecânicas do material, mas podem afetar a estética "
            "e o acabamento."
        ),
        "detection_difficulty": "Moderada — pode confundir-se com sombras ou iluminação irregular"
    },
    "Pitted Surface": {
        "name": "Pitted Surface (Superfície com Cavidades)",
        "description": (
            "Cavidades ou pequenas crateras na superfície do aço, normalmente causadas "
            "por corrosão localizada, erosão ou desgaste mecânico durante o processamento."
        ),
        "causes": [
            "Corrosão localizada por exposição a agentes químicos",
            "Erosão por partículas abrasivas",
            "Desgaste mecânico durante o manuseio"
        ],
        "severity": "Média",
        "severity_reason": (
            "As cavidades reduzem a área efetiva da seção transversal e podem "
            "servir como nucleadores de trincas sob carregamento cíclico."
        ),
        "detection_difficulty": "Moderada — a profundidade das cavidades afeta a detecção"
    },
    "Rolled-in Scale": {
        "name": "Rolled-in Scale (Óxido Laminado)",
        "description": (
            "Óxido (escama) formado durante a laminação a quente que é incorporado à "
            "superfície da chapa, criando uma área áspera e irregular na superfície."
        ),
        "causes": [
            "Formação de óxido na superfície durante o aquecimento",
            "Falha na remoção da escama antes da laminação",
            "Pressão excessiva do laminador que incorpora o óxido"
        ],
        "severity": "Média a Alta",
        "severity_reason": (
            "O óxido incorporado compromete a qualidade superficial e pode causar "
            "falhas em processos de revestimento ou pintura subsequente."
        ),
        "detection_difficulty": "Baixa — geralmente bem visível devido à textura irregular"
    },
    "Scratches": {
        "name": "Scratches (Arranhões)",
        "description": (
            "Arranhões lineares na superfície da chapa de aço, causados por contato "
            "mecânico com equipamentos, roletes danificados ou materiais durante o "
            "manuseio e transporte."
        ),
        "causes": [
            "Contato com roletes danificados ou sujos na linha de laminação",
            "Atrito durante transferência entre equipamentos",
            "Partículas abrasivas no caminho de processamento"
        ],
        "severity": "Média",
        "severity_reason": (
            "Os arranhões não comprometem a integridade estrutural, mas podem afetar "
            "a estética e a adequação para aplicações de acabamento superficial de alta qualidade."
        ),
        "detection_difficulty": "Baixa — geralmente bem visíveis e orientados"
    }
}


def consult_defect(defect_class: str) -> dict:
    """
    Consulta a base de dados sobre um tipo de defeito.

    Args:
        defect_class: Nome da classe do defeito (ex: "Crazing", "Scratches")

    Returns:
        Dicionário com informações completas sobre o defeito.

    Raises:
        ValueError: Se o nome do defeito não for encontrado na base.
    """
    if defect_class not in DEFECT_DATABASE:
        raise ValueError(
            f"Defeito '{defect_class}' não encontrado na base de dados. "
            f"Classes válidas: {list(DEFECT_DATABASE.keys())}"
        )
    return DEFECT_DATABASE[defect_class]


def list_defects() -> list:
    """Retorna a lista de todos os defeitos conhecidos."""
    return [
        {
            "name": v["name"],
            "severity": v["severity"]
        }
        for k, v in DEFECT_DATABASE.items()
    ]


def search_defects_by_severity(severity: str) -> list:
    """
    Busca defeitos por nível de severidade.

    Args:
        severity: Nível de severidade ("Baixa", "Média", "Alta", "Média a Alta")

    Returns:
        Lista de defeitos que correspondem à severidade.
    """
    results = []
    for name, data in DEFECT_DATABASE.items():
        if data["severity"].lower().startswith(severity.lower()):
            results.append({
                "class": name,
                "name": data["name"],
                "severity": data["severity"]
            })
    return results
