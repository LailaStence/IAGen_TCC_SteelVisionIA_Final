# =============================================================================
# SteelVision AI — Módulo de Treinamento da CNN
#
# Contém a definição da arquitetura do modelo e a função de treinamento.
# A arquitetura utiliza 3 camadas convolucionais seguidas de pooling,
# uma camada densa e dropout para regularização, finalizando com
# softmax para classificação de 6 classes.
# =============================================================================

import json
import os
import tensorflow as tf

from utils.dataset_loader import carregar_dataset

# =============================================================================
# Arquitetura do Modelo
# =============================================================================

def criar_modelo():
    """
    Cria e retorna uma CNN para classificação de 6 classes de defeitos.
    
    Arquitetura:
    - Rescaling: normalização 0-1
    - Conv2D(32, 3) -> MaxPooling -> Conv2D(64, 3) -> MaxPooling -> 
      Conv2D(128, 3) -> MaxPooling
    - Flatten -> Dense(256, relu) -> Dropout(0.4) -> Dense(6, softmax)
    """

    modelo = tf.keras.Sequential([

        tf.keras.layers.Rescaling(1. / 255),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(6, activation="softmax")

    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


# =============================================================================
# Função de Treinamento
# =============================================================================

def treinar_modelo(epochs, batch_size, callback, status):
    """
    Treina o modelo CNN com o dataset de defeitos em chapas de aço.

    Args:
        epochs: Número de épocas de treinamento.
        batch_size: Tamanho do lote.
        callback: Função callback para atualização da barra de progresso.
        status: Widget Streamlit para exibir status do treino.

    Returns:
        Historico de treino (dict com accuracy, loss, val_accuracy, val_loss).
    """

    treino, val = carregar_dataset(batch_size)
    modelo = criar_modelo()

    historico_metricas = {
        "accuracy": [],
        "loss": [],
        "val_accuracy": [],
        "val_loss": []
    }

    class Barra(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            callback(int((epoch + 1) / epochs * 100))
            status.text(f"Época {epoch + 1}/{epochs} | "
                        f"Loss: {logs['loss']:.4f} | "
                        f"Val Loss: {logs['val_loss']:.4f}")

            # Armazenar métricas
            historico_metricas["accuracy"].append(logs.get("accuracy", 0))
            historico_metricas["loss"].append(logs.get("loss", 0))
            historico_metricas["val_accuracy"].append(logs.get("val_accuracy", 0))
            historico_metricas["val_loss"].append(logs.get("val_loss", 0))

    modelo.fit(
        treino,
        validation_data=val,
        epochs=epochs,
        callbacks=[Barra()]
    )

    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    modelo.save("models/modelo.keras")

    # Salvar histórico
    historico_dados = {
        "epocas": epochs,
        "batch_size": batch_size,
        "accuracy": historico_metricas["accuracy"],
        "loss": historico_metricas["loss"],
        "val_accuracy": historico_metricas["val_accuracy"],
        "val_loss": historico_metricas["val_loss"]
    }
    with open("models/historico_treino.json", "w") as f:
        json.dump(historico_dados, f, indent=2)

    return historico_metricas
