# Analisador Técnico de Falhas em Aço com IA Generativa

## 🔗 Links Rápidos do Projeto
* **Deploy da Aplicação Web:** [Acesse o SteelVision AI aqui](https://iagentccsteelvisioniafinal-8e8af5bz7qcwqp3ylvqx3y.streamlit.app/)
* **Vídeo de Apresentação (Pitch):** [Assista ao vídeo no YouTube](https://youtu.be/4OpBm-FrQM4)

---

## 1. Descrição do Problema e da Solução
No setor de engenharia de materiais, a análise rápida e precisa de falhas em estruturas de aço é crítica para evitar acidentes e perdas operacionais. Este sistema foi desenvolvido para atuar como um assistente especialista em metalurgia. A IA Generativa é utilizada para interpretar dados técnicos de falhas, processar especificações e acionar ferramentas automatizadas para gerar relatórios e gráficos analíticos de forma imediata.

## 2. Arquitetura do Sistema e Fluxo de LLM
O sistema foi construído utilizando uma arquitetura leve e modular, estruturada da seguinte forma:

## 3. Decisões de Engenharia de LLM e Justificativas

### Escolha do Modelo: Google Gemini (via API)
* **Justificativa:** Optou-se pelo modelo Google Gemini através de chamadas de API comerciais devido ao excelente custo-benefício (janela de cotas gratuitas robusta no Google AI Studio), suporte nativo altamente eficiente para *tool calling* (chamada de ferramentas) em português e baixa latência de resposta.
* **Trade-offs:** Em comparação a um modelo local (como Ollama/Llama 3), a API comercial reduz o custo de hardware local e facilita o deploy, embora dependa de conexão constante com a internet e conformidade com políticas de privacidade de dados em nuvem.

### Framework: SDK Nativo (Sem LangChain/LangGraph)
* **Justificativa:** Escolheu-se a integração direta via SDK oficial do Google para manter a arquitetura da aplicação limpa, leve e com a menor latência possível.
* **Trade-offs:** Evitou-se o uso de frameworks como LangChain para mitigar camadas desnecessárias de abstração e complexity, dado que o escopo do projeto (execução de ferramentas e estruturação de prompts) é perfeitamente atendido de forma nativa.

### Estratégia de Prompting
* **System Prompt:** Isolado no arquivo `prompts/system_prompt.txt` para facilitar o versionamento e a manutenção. Define uma persona especialista em engenharia metalúrgica, com restrições severas de comportamento para evitar alucinações técnicas e instruções explícitas de formato de saída usando tags XML para delimitar raciocínios e dados.

### Configuração de Parâmetros
* **Temperatura (Controle Dinâmico via Slider):** Configurada por padrão em valores baixos (ex: `0.2`) para garantir respostas altamente determinísticas, consistentes e focadas em fatos técnicos. O slider permite a experimentação: valores mais altos aumentam a fluidez do texto, mas elevam o risco de desvios técnicos.
* **Top-P:** Mantido em `0.95` para garantir diversidade controlada de vocabulário sem perder o foco do contexto de engenharia.

### Integração de Ferramentas (Tool Calling)
* **Justificativa:** As funções de análise e geração de gráficos foram isoladas no diretório `tools/`. O modelo decide autonomamente quando acionar essas ferramentas com base nas necessidades do usuário, garantindo que a IA não faça cálculos matemáticos ou gere dados visuais "de cabeça", o que previne alucinações.

## 4. Avaliação de Resultados (O que funcionou e o que não funcionou)
* **O que funcionou:** A separação dos prompts em arquivos `.txt` facilitou drasticamente a engenharia de prompt. O *tool calling* nativo do Gemini demonstrou alta robustez, acionando as funções metalúrgicas com os parâmetros tipados corretos em 100% dos testes realizados em português.
* **O que não funcionou / Limitações:** Em temperaturas superiores a `0.7`, o modelo tentou inventar normas técnicas que não existiam no contexto fornecido. Por isso, travou-se a recomendação operacional do sistema em baixas temperaturas. Como o deploy não exige endpoint público (conforme regras da avaliação), o armazenamento de chaves de API foi centralizado localmente via segredos do Streamlit (`st.secrets`).
