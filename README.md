# Case Técnico de Data Science - iFood

Este repositório contém a solução para o case técnico de Data Science proposto pelo iFood. O objetivo é desenvolver uma solução baseada em dados para otimizar a distribuição de cupons e ofertas aos clientes. A abordagem utiliza análise exploratória, engenharia de features e um modelo de machine learning para prever a probabilidade de um cliente aceitar uma oferta, auxiliando na decisão de qual oferta enviar para cada perfil de cliente.

## 🚀 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir clareza e reprodutibilidade:

```
ifood-case/
├── data/
│   ├── raw/          # Local para os datasets originais (offers.json, etc.)
│   └── processed/    # Local para datasets intermediários ou processados (opcional)
├── notebooks/
│   ├── 1_data_processing.ipynb  # Notebook para análise exploratória e visualização
│   └── 2_modeling.ipynb         # Notebook para engenharia de features, treino e avaliação do modelo
├── src/
│   └── ifood_case/   # Pacote Python com o código fonte modularizado
│       ├── data_processing.py
│       ├── feature_engineering.py
│       ├── model_trainer.py
│       ├── evaluator.py
│       └── ...
├── logs/             # Pasta onde os logs de execução são salvos
├── requirements.txt  # Lista de dependências Python
├── setup.py          # Script para instalar o pacote local 'ifood_case'
└── README.md         # Este arquivo
```

## 🛠️ Stack de Tecnologias

* **Processamento de Dados:** PySpark
* **Análise e Manipulação:** Pandas, NumPy
* **Visualização de Dados:** Seaborn, Matplotlib
* **Modelagem e Machine Learning:** Scikit-learn, LightGBM
* **Otimização de Hiperparâmetros:** Optuna
* **Interpretabilidade do Modelo:** SHAP

## ⚙️ Instalação e Configuração

Siga os passos abaixo para configurar o ambiente e instalar todas as dependências necessárias. É crucial seguir a ordem para garantir que o projeto funcione corretamente.

### Pré-requisitos

* Python 3.8 ou superior
* `pip` e `venv` (geralmente já incluídos com o Python)
* Java Development Kit (JDK) 8 ou superior (requisito para o PySpark)

### Passo a Passo

1.  **Clone o Repositório**

    Navegue até o diretório onde deseja salvar o projeto e clone o repositório do GitHub.

    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd ifood-case
    ```

2.  **Crie e Ative um Ambiente Virtual**

    É uma boa prática isolar as dependências do projeto em um ambiente virtual para evitar conflitos.

    ```bash
    # Criar o ambiente virtual na pasta .venv
    python3 -m venv .venv

    # Ativar o ambiente
    # No macOS ou Linux:
    source .venv/bin/activate
    # No Windows (PowerShell):
    # .\.venv\Scripts\Activate.ps1
    # No Windows (CMD):
    # .\.venv\Scripts\activate.bat
    ```
    Você saberá que o ambiente está ativo pois o seu prompt do terminal mostrará `(.venv)` no início.

3.  **Instale as Dependências**

    Com o ambiente ativado, instale todas as bibliotecas externas listadas no `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Instale o Pacote Local**

    O último passo é instalar o código fonte do diretório `src/` como um pacote Python. A flag `-e` (editável) é importante, pois permite que alterações feitas no código fonte sejam refletidas nos notebooks sem a necessidade de reinstalar.

    ```bash
    pip install -e .
    ```

Pronto! Seu ambiente está configurado.

## 🚀 Como Reproduzir os Experimentos

Para executar a análise e o treinamento do modelo, siga a ordem dos notebooks abaixo.

### 1. Pré-requisito: Dados

Antes de iniciar, certifique-se de que os três datasets brutos fornecidos no case (`offers.json`, `profile.json`, `transactions.json`) estejam localizados dentro da pasta `data/raw/`.

### 2. Execução dos Notebooks

1.  **Inicie o Servidor Jupyter**

    A partir da pasta raiz do projeto no seu terminal (com o ambiente virtual ativado), inicie o Jupyter:
    ```bash
    jupyter notebook
    ```
    ou para uma experiência mais moderna:
    ```bash
    jupyter lab
    ```

2.  **Notebook 1: Análise Exploratória**

    * **Arquivo:** `notebooks/1_data_processing.ipynb`
    * **Propósito:** Este notebook carrega os dados brutos, utiliza a classe `DataVisualizer` para realizar uma análise exploratória completa (univariada e multivariada) e gera os principais insights visuais sobre o comportamento dos clientes e a performance das ofertas.

3.  **Notebook 2: Modelagem e Avaliação**

    * **Arquivo:** `notebooks/2_modeling.ipynb`
    * **Propósito:** Este notebook orquestra o pipeline de Machine Learning. Ele utiliza a classe `FeatureEngineering` para criar um dataset robusto, treina um modelo de propensão (LightGBM) com otimização de hiperparâmetros (Optuna), e por fim utiliza a classe `Evaluator` para avaliar a performance do modelo com métricas técnicas (AUC, Matriz de Confusão) e de negócio (Uplift Financeiro).
