"""Streamlit Web Application for the iFood Customer Offer Propensity Model.

This application provides a user interface to:
1. Trigger the end-to-end model training pipeline.
2. Make real-time predictions on single instances using a trained model.
"""

import logging
import re
import sys

# Adiciona o diretório 'src' ao path para que possamos importar os módulos do projeto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from pyspark.sql import SparkSession

from ifood_case.logger_config import setup_logging
from ifood_case.pipeline import TrainingPipeline
from ifood_case.utils import create_cyclical_features_pandas

project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- CONFIGURAÇÃO DA PÁGINA E LOGGING ---
st.set_page_config(page_title="iFood Offer Propensity", layout="wide")
setup_logging()
logger = logging.getLogger(__name__)


# --- FUNÇÕES AUXILIARES ---


@st.cache_resource
def get_spark_session():
    """Creates and returns a SparkSession."""
    return (
        SparkSession.builder.appName("iFoodStreamlitApp")
        .config("spark.driver.memory", "4g")  # Example: 4 gigabytes for the driver
        .config("spark.executor.memory", "8g")  # Example: 8 gigabytes for each executor
        .getOrCreate()
    )


def plot_shap_force_plot(classifier, preprocessed_data: np.ndarray, features_names: list[str]):
    """Calculates and renders a SHAP force plot for a single prediction."""
    st.subheader("Análise da Predição (SHAP Force Plot)")
    st.markdown(
        "Este gráfico mostra quais características mais influenciaram a predição. "
        "**Forças em vermelho** aumentam a probabilidade de sucesso, enquanto **forças em azul** a diminuem.",
    )

    preprocessed_data_df = pd.DataFrame(preprocessed_data, columns=features_names)

    # Usamos o TreeExplainer otimizado para o LightGBM
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(preprocessed_data_df)

    # Plotamos para a classe positiva (1) e para a primeira (e única) amostra
    shap.plots.waterfall(explanation[0,:],max_display=15)

    # Usamos st.pyplot() para renderizar o gráfico matplotlib no Streamlit
    st.pyplot(bbox_inches="tight")
    plt.close()  # Limpa a figura para evitar sobreposição em execuções futuras


def get_available_models(models_path: Path) -> list:
    """Scans the models directory for trained model files."""
    if not models_path.exists():
        return []

    # Extrai o GUID dos nomes dos arquivos para agrupar modelo e calibrador
    model_guids = set()
    for f in models_path.iterdir():
        if f.is_file() and f.suffix == ".joblib":
            # Extrai o GUID usando regex
            match = re.search(r"_([a-f0-9\-]{36})\.joblib", f.name)
            if match:
                model_guids.add(match.group(1))
    return sorted(list(model_guids), reverse=True)


# --- INTERFACE PRINCIPAL ---

st.title("iFood - Modelo de Propensão a Ofertas")
st.markdown("Esta aplicação permite treinar o modelo de Machine Learning e fazer predições em tempo real.")

# --- SEÇÃO DE TREINAMENTO ---
st.header("1. Treinamento do Modelo")
st.markdown(
    "Clique no botão abaixo para executar o pipeline completo. Isso irá carregar os dados brutos, "
    "realizar a engenharia de features, treinar, calibrar e salvar um novo modelo. "
    "**Atenção:** Este processo pode ser demorado.",
)

if st.button("Iniciar Treinamento Completo"):
    with st.spinner("Executando o pipeline de treinamento... Por favor, aguarde."):
        try:
            st.info("Iniciando SparkSession...")
            spark = get_spark_session()

            spark = (
                SparkSession.builder.appName("IfoodTrainingRun")
                .config("spark.driver.memory", "4g")  # Example: 4 gigabytes for the driver
                .config("spark.executor.memory", "8g")  # Example: 8 gigabytes for each executor
                .getOrCreate()
            )
            spark.sparkContext.setLogLevel("ERROR")

            st.info("Instanciando e executando o pipeline de treino...")
            pipeline = TrainingPipeline(spark=spark)
            pipeline.train()  # O método já salva os modelos

            st.success("Pipeline de treinamento concluído com sucesso!")
            st.info("Um novo modelo foi treinado e salvo na pasta 'models/'.")
            st.balloons()
        except Exception as e:
            st.error(f"Ocorreu um erro durante o treinamento: {e}")
            logger.error("Erro no pipeline de treinamento via Streamlit", exc_info=True)


# --- SEÇÃO DE PREDIÇÃO ---
st.header("2. Fazer uma Predição")
st.markdown("Use um modelo treinado para prever a probabilidade de sucesso de uma oferta para um novo cliente.")

models_dir = project_root / "models"
available_guids = get_available_models(models_dir)

if not available_guids:
    st.warning("Nenhum modelo treinado foi encontrado. Por favor, execute o treinamento primeiro.")
else:
    # --- SELEÇÃO DO MODELO ---
    selected_guid = st.selectbox("Selecione a versão do modelo para usar:", available_guids)

    st.info(f"Modelo selecionado: `{selected_guid}`")

    # --- INPUTS DO USUÁRIO ---
    st.subheader("Insira os dados da oferta e do cliente:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Características da Oferta**")
        offer_type = st.selectbox("Tipo da Oferta (offer_type)", ["bogo", "discount", "informational"])
        last_offer_type = st.selectbox("Última Oferta (last_offer_type)", ["bogo", "discount", "informational", None])
        duration = st.slider("Duração (dias)", 1, 10, 7)
        min_value = st.slider("Valor Mínimo ($)", 0, 20, 10)
        discount_value = st.slider("Valor do Desconto ($)", 0, 20, 5)
        offers_viewed_count_before = st.number_input(" Total De Ofertas Já Vistas", value=5, step=1)
        offers_completed_count_before = st.number_input(" Total De Ofertas Já Utilizadas", value=3, step=1)

    with col2:
        st.write("**Histórico do Cliente (Point-in-Time)**")
        total_spend_before = st.number_input("Gasto Total Anterior ($)", value=150.0, step=10.0)
        avg_ticket_before = st.number_input("Ticket Médio Anterior ($)", value=30.0, step=5.0)
        customer_conversion_rate_before = st.slider("Taxa de Conversão Anterior (%)", 0.0, 100.0, 50.0)
        max_ticket_before = st.number_input("Ticket Máximo Anterior ($)", value=70.0, step=5.0)
        min_ticket_before = st.number_input("Ticket Mínimo Anterior ($)", value=10.0, step=5.0)
        email = int(st.checkbox("Chegou por Email"))
        web = int(st.checkbox("Chegou por Web"))
        social = int(st.checkbox("Chegou por Social"))
        mobile = int(st.checkbox("Chegou por Mobile"))

    with col3:
        st.write("**Perfil do Cliente**")
        gender = st.selectbox("Gênero", ["M", "F", "O", "Unknown"])
        age = st.slider("Idade", 18, 100, 45)
        credit_card_limit = st.slider("Limite do Cartão de Crédito", 1000, 120000, 50000)
        registered_date = str(st.date_input("Quando se registrou")).replace("-", "")

    if st.button("Calcular Probabilidade de Sucesso"):
        # --- LÓGICA DE PREDIÇÃO ---
        with st.spinner("Processando a predição..."):
            try:
                # Criar um DataFrame Pandas com os inputs
                # Nota: Este é um exemplo simplificado. O modelo real espera TODAS as features.
                # Para uma aplicação real, você precisaria de todos os inputs.
                # Aqui, usamos um dicionário com valores padrão para as features que não estão na UI.
                input_data_dict = {
                    "total_spend_before": [total_spend_before],
                    "transaction_count_before": [5],  # Exemplo
                    "registered_date": [registered_date],
                    "avg_ticket_before": [avg_ticket_before],
                    "max_ticket_before": [max_ticket_before],  # Exemplo
                    "min_ticket_before": [min_ticket_before],  # Exemplo
                    "offers_viewed_count_before": [offers_viewed_count_before],  # Exemplo
                    "offers_completed_count_before": [offers_completed_count_before],  # Exemplo
                    "customer_conversion_rate_before": [customer_conversion_rate_before],
                    "age": [age],
                    "credit_card_limit": [credit_card_limit],
                    "duration": [duration],
                    "min_value": [min_value],
                    "discount_value": [discount_value],
                    "reward_ratio": [discount_value / min_value if min_value > 0 else 0],
                    "channel_is_mobile": [mobile],  # Exemplo
                    "channel_is_email": [email],  # Exemplo
                    "channel_is_social": [social],  # Exemplo
                    "channel_is_web": [web],  # Exemplo
                    "last_offer_viewed_type": last_offer_type,
                    "offer_type": offer_type,
                    "gender": gender,
                }

                input_df = pd.DataFrame(input_data_dict)
                input_df = create_cyclical_features_pandas(input_df, "registered_date")

                # Chamar o método estático de predição
                probabilities, classifier, input_transformed, features_names = TrainingPipeline.predict(
                    input_data=input_df,
                    model_id=selected_guid,
                )

                # Exibir o resultado
                success_probability = probabilities[0, 1]

                st.success("Predição calculada com sucesso!")
                st.metric(
                    label="Probabilidade de Sucesso da Oferta",
                    value=f"{success_probability:.2%}",
                )

                st.progress(success_probability)

                plot_shap_force_plot(classifier, input_transformed, features_names)

            except Exception as e:
                st.error(f"Ocorreu um erro durante a predição: {e}")
                logger.error("Erro na predição via Streamlit", exc_info=True)
