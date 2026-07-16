import tensorflow as tf

IMG_SIZE = (200,200)

def carregar_dataset(batch):

    treino = tf.keras.utils.image_dataset_from_directory(

        r"C:\Users\Usuario\Desktop\UniSENAI_IAI\IA_Generativa\dataset\NEU-DET\train\images",

        image_size=IMG_SIZE,

        batch_size=batch

    )

    validacao = tf.keras.utils.image_dataset_from_directory(

        r"C:\Users\Usuario\Desktop\UniSENAI_IAI\IA_Generativa\dataset\NEU-DET\validation\images",

        image_size=IMG_SIZE,

        batch_size=batch

    )

    return treino,validacao
