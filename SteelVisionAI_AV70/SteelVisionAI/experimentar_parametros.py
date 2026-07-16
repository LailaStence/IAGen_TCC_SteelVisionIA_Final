# =============================================================================
# SteelVision AI — Experimentação de Parâmetros do LLM
#
# Script para testar diferentes configurações de temperatura, top-p e modelos,
# documentando como cada configuração afeta o comportamento da geração.
#
# Uso:
#   python experimentar_parametros.py
#   python experimentar_parametros.py --temperatura 0.5 --modelo gpt-4o
# =============================================================================

import argparse
import json
import os
from datetime import datetime

# =============================================================================
# CONFIGURAÇÕES TESTADAS
# =============================================================================

EXPERIMENTOS = [
    {
        "nome": "Determinístico (Temp=0.0)",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 500,
        "descricao": "Temperatura 0 força o modelo a sempre escolher o token mais provável. Resultado determinístico e repetível."
    },
    {
        "nome": "Conservador (Temp=0.3)",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 500,
        "descricao": "Baixa temperatura com top-p restrito. Relatórios técnicos precisos mas menos variados."
    },
    {
        "nome": "Equilibrado (Temp=0.7)",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 500,
        "descricao": "Equilíbrio entre precisão técnica e fluidez. Escolha padrão do sistema."
    },
    {
        "nome": "Criativo (Temp=1.0)",
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 500,
        "descricao": "Alta temperatura permite mais variação. Risco de gerar recomendações genéricas."
    },
    {
        "nome": "Top-p Restrito (Temp=0.7, Top-p=0.5)",
        "temperature": 0.7,
        "top_p": 0.5,
        "max_tokens": 500,
        "descricao": "Nucleus sampling mais restritivo. Reduz alucinações mas limita vocabulário."
    }
]

# Defeitos de teste
DEFEITOS_TESTE = ["Crazing", "Scratches", "Inclusion"]


def _try_llm_call(config, defect_class):
    """
    Tenta fazer uma chamada ao LLM com a configuração especificada.

    Returns:
        Texto da resposta ou None se falhar.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        client = OpenAI(base_url=api_base)

        # Carregar system prompt
        system_prompt = ""
        path = os.path.join("prompts", "system_prompt.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                system_prompt = f.read().strip()

        # Carregar few-shot
        few_shot = ""
        path = os.path.join("prompts", "few_shot_examples.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                few_shot = f.read().strip()

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if few_shot:
            messages.append({"role": "assistant", "content": few_shot})

        messages.append({
            "role": "user",
            "content": (
                f"Análise de defeito detectado:\n"
                f"- Classe: {defect_class}\n"
                f"- Confiança: 92.5%\n"
                f"Gere o relatório técnico."
            )
        })

        response = client.chat.completions.create(
            model=config.get("model", "gpt-4o-mini"),
            messages=messages,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"  Erro: {e}")
        return None


def _generate_local_report(defect_class):
    """Gera relatório local (fallback sem LLM)."""
    from agents.quality_assistant import QualityAssistant
    agent = QualityAssistant()
    analysis = agent.analyze(defect_class, 92.5)
    return agent.generate_text_report(analysis)


def main():
    parser = argparse.ArgumentParser(
        description="Experimentar parâmetros do LLM no SteelVision AI."
    )
    parser.add_argument("--temperatura", type=float, default=None,
                        help="Temperatura específica para teste único")
    parser.add_argument("--modelo", type=str, default=None,
                        help="Modelo específico para teste único")
    args = parser.parse_args()

    print("=" * 70)
    print("  SteelVision AI — Experimentação de Parâmetros LLM")
    print("=" * 70)
    print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Teste único ou todos os experimentos
    if args.temperatura is not None:
        config_test = {
            "nome": f"Teste personalizado (Temp={args.temperatura})",
            "temperature": args.temperatura,
            "top_p": 0.9,
            "max_tokens": 500,
            "model": args.modelo or "gpt-4o-mini",
            "descricao": f"Temperatura={args.temperatura}, Modelo={args.modelo or 'gpt-4o-mini'}"
        }
        configs = [config_test]
    else:
        configs = EXPERIMENTOS

    resultados = []

    for config in configs:
        print(f"\n{'─' * 70}")
        print(f"  Experimento: {config['nome']}")
        print(f"  Temperatura: {config['temperature']} | Top-p: {config['top_p']} | "
              f"Max Tokens: {config['max_tokens']}")
        print(f"  Descrição: {config['descricao']}")
        print(f"{'─' * 70}")

        for defeito in DEFEITOS_TESTE:
            print(f"\n  Defeito: {defeito}")
            print(f"  Tentando LLM...")

            llm_response = _try_llm_call(config, defeito)

            if llm_response:
                print(f"  Status: OK (LLM)")
                word_count = len(llm_response.split())
                print(f"  Tokens estimados: ~{word_count} palavras")
                print(f"  Preview (100 chars): {llm_response[:100]}...")
                resultado = {
                    "experimento": config["nome"],
                    "defeito": defeito,
                    "fonte": "llm",
                    "config": config,
                    "word_count": word_count,
                    "preview": llm_response[:200]
                }
            else:
                print(f"  Status: Fallback local (LLM não disponível)")
                local_report = _generate_local_report(defeito)
                word_count = len(local_report.split())
                resultado = {
                    "experimento": config["nome"],
                    "defeito": defeito,
                    "fonte": "local",
                    "config": config,
                    "word_count": word_count,
                    "preview": local_report[:200]
                }

            resultados.append(resultado)

    # Salvar resultados
    output = {
        "data": datetime.now().isoformat(),
        "resultados": resultados,
        "configuracoes_testadas": [
            {"nome": c["nome"], "temperature": c["temperature"], "top_p": c["top_p"]}
            for c in configs
        ]
    }

    os.makedirs("models", exist_ok=True)
    output_path = "models/resultados_experimentos.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  Resultados salvos em: {output_path}")
    print(f"  Total de experimentos: {len(resultados)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
