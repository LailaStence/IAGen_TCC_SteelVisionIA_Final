import os
import shutil
import random

random.seed(42)

ORIGEM = r"SteelVisionAI\dataset\train\images"

DESTINO_TREINO = "dataset/train"

DESTINO_VALIDACAO = "dataset/validation"

PORCENTAGEM_VALIDACAO = 0.20


classes = [

    pasta

    for pasta in os.listdir(ORIGEM)

    if os.path.isdir(os.path.join(ORIGEM,pasta))

]


for classe in classes:

    origem = os.path.join(ORIGEM,classe)

    treino = os.path.join(DESTINO_TREINO,classe)

    validacao = os.path.join(DESTINO_VALIDACAO,classe)

    os.makedirs(treino,exist_ok=True)

    os.makedirs(validacao,exist_ok=True)

    imagens = [

        img

        for img in os.listdir(origem)

        if img.lower().endswith(

            (".jpg",".jpeg",".png",".bmp")

        )

    ]

    random.shuffle(imagens)

    limite = int(len(imagens)*PORCENTAGEM_VALIDACAO)

    validacao_imgs = imagens[:limite]

    treino_imgs = imagens[limite:]

    for img in treino_imgs:

        shutil.copy2(

            os.path.join(origem,img),

            os.path.join(treino,img)

        )

    for img in validacao_imgs:

        shutil.copy2(

            os.path.join(origem,img),

            os.path.join(validacao,img)

        )

print("Dataset organizado com sucesso!")
