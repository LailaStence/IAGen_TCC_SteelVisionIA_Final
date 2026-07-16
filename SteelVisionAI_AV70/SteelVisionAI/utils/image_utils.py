import numpy as np

from PIL import Image


IMG_SIZE = 200


def preprocessar(imagem):

    imagem = imagem.convert("RGB")

    imagem = imagem.resize(

        (IMG_SIZE, IMG_SIZE)

    )

    imagem = np.array(imagem)

    imagem = imagem.astype("float32")

    imagem = imagem / 255

    imagem = np.expand_dims(

        imagem,

        axis=0

    )

    return imagem
