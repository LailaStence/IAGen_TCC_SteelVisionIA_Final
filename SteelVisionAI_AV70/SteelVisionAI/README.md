# SteelVision AI v2.0

**Sistema de Visão Computacional integrado com IA Generativa para Inspeção de Qualidade**

O SteelVision AI é um sistema inteligente desenvolvido para automatizar a inspeção de qualidade na indústria siderúrgica. Na versão 2.0, o sistema integra uma Rede Neural Convolucional (CNN) para classificação visual com um Modelo de Linguagem Grande (LLM) para geração de relatórios técnicos contextuais.

Este projeto foi desenvolvido como complemento do Trabalho de Conclusão de Curso (TCC) pela aluna **Laila Michely Stence**, no curso de Pós-Graduação em Inteligência Artificial Aplicada.

---

## 1. Descrição do Problema e da Solução

Na indústria siderúrgica, a inspeção visual manual de chapas de aço é um processo demorado e sujeito a erros. Defeitos superficiais podem comprometer a integridade do material. O sistema original resolvia apenas a classificação visual (identificar o defeito). 

**A Solução com IA Generativa:** A versão 2.0 resolve o problema de contextualização. A CNN identifica o defeito, e o LLM atua como um "Engenheiro Assistente", gerando relatórios técnicos completos com causas prováveis, ações corretivas e medidas preventivas, consultando ferramentas estruturadas para evitar alucinações.

## 2. Arquitetura do LLM

A arquitetura do LLM integra o fluxo da aplicação da seguinte forma:

**Fluxo de Dados:**
`Input do Usuário (Imagem)` → `CNN (Classificação)` → `Agente de Qualidade (Orquestrador)` → `LLM (com Tool Calling)` → `Relatório Técnico`

| Componente | Framework / Implementação | Justificativa |
| :--- | :--- | :--- |
| **Framework** | Chamadas diretas à API (OpenAI-compatible) | Escolhido pela simplicidade e controle total sobre o fluxo de tool calling, sem a sobrecarga de abstrações do LangChain. |
| **Modelo** | GPT-4o-mini | Modelo leve, rápido e de baixo custo, ideal para gerar relatórios estruturados sem necessidade de raciocínio extremamente complexo. |
| **Agente** | Single-turn com Tool Calling | O agente não mantém um loop de conversa complexo, apenas orquestra as ferramentas para o LLM gerar a resposta final. |

## 3. Decisões e Justificativas

### 3.1. Escolha do Modelo e Framework
Optamos por **chamadas diretas à API** ao invés de frameworks como LangChain. A estrutura de ferramentas é simples (3 tools de leitura de bases JSON), e a abstração do LangChain não traria benefícios claros, apenas complexidade. O modelo **GPT-4o-mini** foi escolhido pelo excelente custo-benefício e robustez em formatação estruturada (JSON/Markdown).

### 3.2. System Prompt e Estratégias de Prompting
O System Prompt define a persona do LLM como um "Engenheiro de Qualidade Siderúrgico". As regras de formatação são estritas, exigindo seções claras.
* **Tool Calling (Funções):** O LLM é forçado a usar as ferramentas `consult_defect`, `get_corrections` e `get_prediction_statistics` para obter dados reais, eliminando a alucinação.
* **Few-Shot Prompting:** O prompt inclui dois exemplos completos de relatórios (Exemplo 1 para Scratches, Exemplo 2 para Crazing) para ensinar o formato exato esperado.

### 3.3. Parâmetros Configurados
A configuração do LLM foi experimentada e ajustada para o caso de uso técnico:
* **Temperatura (0.7):** Equilíbrio entre precisão técnica e fluidez na escrita. Valores mais altos (1.0) geravam recomendações genéricas demais.
* **Top-p (0.9):** Nucleus sampling para limitar alucinações, mantendo um vocabulário natural.
* **Max Tokens (500):** Limite para garantir relatórios concisos e objetivos.

## 4. O que Funcionou

A integração do **Tool Calling** foi a decisão que mais agregou valor. O LLM parou de inventar causas e correções, passando a consultar rigorosamente as bases de dados locais (`tools/defect_database.py` e `tools/correction_guide.py`). A estratégia de **Few-Shot** funcionou perfeitamente para padronizar a saída em Markdown.

## 5. O que NÃO Funcionou e Limitações

* **Dependência de API:** Se a chave de API (OpenAI) não estiver disponível, o sistema faz fallback para o `QualityAssistant` local (que gera o relatório sem LLM). O prompt de fallback foi ajustado, mas perde a fluidez gerativa.
* **Latência:** A chamada à API do LLM adiciona um atraso de 2-4 segundos ao processo de classificação, o que pode ser percebido pelo usuário final.

---

## 6. Estrutura do Projeto

A estrutura do repositório foi organizada para facilitar a localização dos artefatos de engenharia de LLM:

```text
SteelVisionAI/
├── prompts/                     # Prompts e templates do LLM
│   ├── system_prompt.txt        # System prompt principal
│   ├── report_prompt.txt        # Prompt de geração de relatórios
│   ├── suggestion_prompt.txt    # Prompt para sugestões estruturadas
│   └── few_shot_examples.txt    # Exemplos few-shot para o LLM
├── tools/                       # Ferramentas tipadas para o LLM
│   ├── defect_database.py       # Consulta informações de defeitos
│   ├── correction_guide.py      # Consulta ações corretivas
│   └── statistics_tool.py       # Consulta estatísticas do dataset
├── agents/                      # Lógica dos agentes
│   ├── quality_assistant.py     # Agente orquestrador (fallback local)
│   └── report_generator.py      # Agente gerador de relatórios (LLM)
├── app.py                       # Aplicação principal (UI + Integração)
├── pages/                       # Páginas secundárias (Streamlit)
├── utils/                       # Utilitários (CNN, Dataset)
└── README.md
```

---

### Referências

[1] Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. *Applied Surface Science*, 285, 858-864. Dataset disponível em: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
