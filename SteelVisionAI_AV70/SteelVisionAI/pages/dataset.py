import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

st.set_page_config(page_title="Dataset", page_icon="📂")

st.title("📂 Dataset")

TRAIN_PATH = r"C:\Users\Usuario\Desktop\UniSENAI_IAI\IA_Generativa\dataset\NEU-DET\train\images"
VAL_PATH = "dataset/validation"


def contar_imagens(caminho):

    dados = {}

    if not os.path.exists(caminho):
        return dados

    for classe in sorted(os.listdir(caminho)):

        pasta = os.path.join(caminho,classe)

        if os.path.isdir(pasta):

            arquivos = [

                f for f in os.listdir(pasta)

                if f.lower().endswith(
                    (".jpg",".jpeg",".png",".bmp")
                )

            ]

            dados[classe]=len(arquivos)

    return dados


train = contar_imagens(TRAIN_PATH)

validation = contar_imagens(VAL_PATH)

st.subheader("Quantidade de imagens")

df = pd.DataFrame({

    "Treino":train,

    "Validação":validation

}).fillna(0)

st.dataframe(df)

st.divider()

st.subheader("Distribuição das classes")

fig,ax = plt.subplots(figsize=(9,4))

df.plot.bar(ax=ax)

plt.xticks(rotation=30)

plt.ylabel("Quantidade")

st.pyplot(fig)

st.divider()

st.subheader("Exemplo de imagens")

colunas = st.columns(3)

indice = 0

for classe in train.keys():

    pasta = os.path.join(TRAIN_PATH,classe)

    imagens = [

        f for f in os.listdir(pasta)

        if f.endswith(
            (".jpg",".png",".bmp",".jpeg")
        )

    ]

    if len(imagens)==0:
        continue

    img = random.choice(imagens)

    imagem = Image.open(os.path.join(pasta,img))

    with colunas[indice]:

        st.image(imagem,use_container_width=True)

        st.caption(classe)

    indice +=1

    if indice==3:
        indice=0

st.success("Dataset carregado com sucesso.")
