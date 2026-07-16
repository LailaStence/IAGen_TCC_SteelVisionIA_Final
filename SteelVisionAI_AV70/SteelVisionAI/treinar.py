# =============================================================================
# SteelVision AI — Script de Treinamento da CNN
#
# Script de linha de comando para treinar o modelo de classificação de defeitos
# em chapas de aço. Utiliza a arquitetura CNN definida em utils/trainer.py.
#
# Uso:
#   python treinar.py
#   python treinar.py --epochs 30 --batch-size 32
# =============================================================================

import argparse
import json
import os
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

from utils.dataset_loader import carregar_dataset
from utils.trainer import criar_modelo

# =============================================================================
# Configuração
# =============================================================================

IMG_SIZE = 200
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

MODEL_SAVE_PATH = "models/modelo.keras"
HISTORY_SAVE_PATH = "models/historico_treino.json"


def salvar_historico(historico, epocas, batch_size):
    """Salva o histórico de treino em formato JSON."""
    os.makedirs("models", exist_ok=True)
    dados = {
        "epocas": epocas,
        "batch_size": batch_size,
        "accuracy": historico.history["accuracy"],
        "loss": historico.history["loss"],
        "val_accuracy": historico.history["val_accuracy"],
        "val_loss": historico.history["val_loss"]
    }
    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(dados, f, indent=2)
    print(f"Histórico salvo em {HISTORY_SAVE_PATH}")


def gerar_grafico(historico):
    """Gera gráficos de acurácia e loss do treino."""
    os.makedirs("models", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Acurácia
    ax1.plot(historico.history["accuracy"], "b-", label="Treino")
    ax1.plot(historico.history["val_accuracy"], "r-", label="Validação")
    ax1.set_title("Acurácia")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Acurácia")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(historico.history["loss"], "b-", label="Treino")
    ax2.plot(historico.history["val_loss"], "r-", label="Validação")
    ax2.set_title("Função de Perda (Loss)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/curvas_treino.png", dpi=150)
    plt.close()
    print("Gráficos salvos em models/curvas_treino.png")


def main():
    parser = argparse.ArgumentParser(
        description="Treinar modelo CNN para classificação de defeitos em chapas de aço."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Número de épocas de treinamento (padrão: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote (batch size) (padrão: 32)"
    )
    args = parser.parse_args()

    epocas = args.epochs
    batch_size = args.batch_size

    print("=" * 60)
    print("  SteelVision AI — Treinamento do Modelo CNN")
    print("=" * 60)
    print(f"  Épocas: {epocas}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Tamanho da imagem: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Classes: {len(CLASSES)}")
    print("=" * 60)

    # Carregar dataset
    print("\n[1/4] Carregando dataset...")
    treino, validacao = carregar_dataset(batch_size)
    print("  Dataset carregado com sucesso.")

    # Criar modelo
    print("[2/4] Criando modelo CNN...")
    modelo = criar_modelo()
    modelo.summary()

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Treinar
    print(f"[3/4] Iniciando treinamento ({epocas} épocas)...")
    historico = modelo.fit(
        treino,
        validation_data=validacao,
        epochs=epocas,
        callbacks=[early_stopping]
    )

    # Salvar modelo
    print("[4/4] Salvando modelo...")
    os.makedirs("models", exist_ok=True)
    modelo.save(MODEL_SAVE_PATH)
    print(f"  Modelo salvo em {MODEL_SAVE_PATH}")

    # Métricas finais
    acc = historico.history["accuracy"][-1]
    val_acc = historico.history["val_accuracy"][-1]
    loss = historico.history["loss"][-1]
    val_loss = historico.history["val_loss"][-1]

    print("\n" + "=" * 60)
    print("  RESULTADOS FINAIS")
    print("=" * 60)
    print(f"  Acurácia (Treino):       {acc:.4f}")
    print(f"  Acurácia (Validação):    {val_acc:.4f}")
    print(f"  Loss (Treino):           {loss:.4f}")
    print(f"  Loss (Validação):        {val_loss:.4f}")
    print("=" * 60)

    # Salvar histórico e gráficos
    salvar_historico(historico, epocas, batch_size)
    gerar_grafico(historico)

    print("\nTreinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
