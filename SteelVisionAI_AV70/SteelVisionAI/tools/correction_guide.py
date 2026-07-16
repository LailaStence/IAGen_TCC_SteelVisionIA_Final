# =============================================================================
# SteelVision AI — Ferramenta: Guia de Correções
#
# Tool disponibilizada ao LLM para consultar ações corretivas e preventivas
# para cada tipo de defeito em chapas de aço.
#
# Justificativa: O LLM precisa de um guia estruturado para sugerir ações
# realistas e baseadas em práticas da indústria siderúrgica, não inventar
# recomendações genéricas.
# =============================================================================

import json

# Guia de ações corretivas e preventivas para cada defeito
# Baseado em práticas da indústria siderúrgica

CORRECTION_GUIDE = {
    "Crazing": {
        "immediate_actions": [
            "Revisar o perfil de resfriamento controlado pós-laminação",
            "Implementar tratamento térmico de alívio de tensões no lote",
            "Separar o lote para avaliação complementar por ensaios não destrutivos"
        ],
        "preventive_measures": [
            "Calibrar o sistema de resfriamento a cada turno de produção",
            "Monitorar continuamente a temperatura de saída do laminador",
            "Realizar análise química de cada lote antes da laminação"
        ],
        "process_adjustments": [
            "Reduzir a velocidade de resfriamento em 10-15%",
            "Ajustar a distribuição de água nos sistemas de resfriamento"
        ],
        "estimated_cost_impact": "Moderado — envolve calibração e monitoramento"
    },
    "Inclusion": {
        "immediate_actions": [
            "Verificar a pureza do metal líquido antes da laminação",
            "Aumentar a frequência de análise de inclusões no metal",
            "Implementar filtragem adicional do aço líquido"
        ],
        "preventive_measures": [
            "Melhorar o processo de desoxidação na aciaria",
            "Implementar controle mais rigoroso dos aditivos",
            "Revisar o processo de escorificação do aço"
        ],
        "process_adjustments": [
            "Ajustar a quantidade de desoxidantes utilizados",
            "Otimizar o tempo de repouso do metal antes da laminação"
        ],
        "estimated_cost_impact": "Moderado a Alto — pode exigir investimento em filtragem"
    },
    "Patches": {
        "immediate_actions": [
            "Identificar a origem da contaminação na superfície",
            "Limpar a área afetada e avaliar extensão do dano",
            "Verificar a qualidade dos fluidos de processamento"
        ],
        "preventive_measures": [
            "Implementar controle de qualidade dos lubrificantes",
            "Instalar barreiras de proteção contra contaminação",
            "Programar limpeza preventiva dos equipamentos a cada 8 horas"
        ],
        "process_adjustments": [
            "Ajustar a concentração de fluidos de resfriamento",
            "Verificar a compatibilidade dos lubrificantes com o aço"
        ],
        "estimated_cost_impact": "Baixo — ajustes operacionais simples"
    },
    "Pitted Surface": {
        "immediate_actions": [
            "Identificar a fonte de corrosão ou erosão",
            "Avaliar a profundidade das cavidades com micrômetro",
            "Verificar o estado dos equipamentos de processamento"
        ],
        "preventive_measures": [
            "Implementar sistema de proteção contra corrosão",
            "Melhorar o controle ambiental da área de processamento",
            "Substituir componentes desgastados dos equipamentos"
        ],
        "process_adjustments": [
            "Ajustar a umidade ambiente na área de laminação",
            "Otimizar o sistema de ventilação e exaustão"
        ],
        "estimated_cost_impact": "Moderado — pode exigir substituição de componentes"
    },
    "Rolled-in Scale": {
        "immediate_actions": [
            "Verificar o sistema de remoção de escama (descaling)",
            "Ajustar a pressão do jato de água de descaling",
            "Inspeccionar os rolos do laminador quanto a resíduos"
        ],
        "preventive_measures": [
            "Manter o sistema de descaling em funcionamento contínuo",
            "Programar manutenção preventiva dos bicos de jato de água",
            "Monitorar a temperatura de aquecimento para controlar a formação de óxido"
        ],
        "process_adjustments": [
            "Aumentar a pressão dos jatos de descaling em 5-10%",
            "Reduzir o tempo de exposição ao ar durante o aquecimento"
        ],
        "estimated_cost_impact": "Moderado — manutenção do sistema de descaling"
    },
    "Scratches": {
        "immediate_actions": [
            "Inspeccionar e substituir roletes danificados ou sujos",
            "Limpar toda a linha de transporte entre estações",
            "Verificar o alinhamento dos rolos de transporte"
        ],
        "preventive_measures": [
            "Programar manutenção preventiva mensal dos roletes",
            "Instalar sensores de vibração para detecção precoce de desalinhamentos",
            "Implementar barreira de proteção entre estações de transporte"
        ],
        "process_adjustments": [
            "Ajustar a tensão nos rolos de transporte",
            "Otimizar a velocidade de transferência entre equipamentos"
        ],
        "estimated_cost_impact": "Baixo — manutenção básica de equipamentos"
    }
}


def get_corrections(defect_class: str) -> dict:
    """
    Consulta o guia de correções para um tipo de defeito.

    Args:
        defect_class: Nome da classe do defeito (ex: "Crazing", "Scratches")

    Returns:
        Dicionário com ações imediatas, preventivas, ajustes de processo e impacto de custo.

    Raises:
        ValueError: Se o nome do defeito não for encontrado no guia.
    """
    if defect_class not in CORRECTION_GUIDE:
        raise ValueError(
            f"Defeito '{defect_class}' não encontrado no guia de correções. "
            f"Classes válidas: {list(CORRECTION_GUIDE.keys())}"
        )
    return CORRECTION_GUIDE[defect_class]


def list_corrections_summary() -> list:
    """Retorna resumo das correções para todos os defeitos."""
    return [
        {
            "defect": k,
            "immediate_count": len(v["immediate_actions"]),
            "preventive_count": len(v["preventive_measures"]),
            "cost_impact": v["estimated_cost_impact"]
        }
        for k, v in CORRECTION_GUIDE.items()
    ]
