#############################
#   Autorzy:
#   Kajetan Frąckowiak s28404
#   Marek Walkowski    s25378
#############################
import matplotlib.pyplot as plt
import json
import os

if not os.path.exists("plots"):
    os.makedirs("plots")


def plot_models_hidden_with_hidden_64():
    """Plot training loss for models with hidden size 64."""
    with open("disease_loss_hidden_size_64_iter_200.json", "r") as f:
        disease_data = json.load(f)
    with open("iris_loss_hidden_size_64_iter_200.json", "r") as f:
        iris_data = json.load(f)
    with open("cifar10_loss_hidden_size_64_iter_200.json", "r") as f:
        cifar10_data = json.load(f)
    with open("fashion_mnist_loss_hidden_size_64_iter_200.json", "r") as f:
        fashion_mnist_data = json.load(f)

    plt.plot(disease_data["loss"], label="Heart Disease UCI")
    plt.plot(iris_data["loss"], label="Fisher Iris")
    plt.plot(cifar10_data["loss"], label="CIFAR-10")
    plt.plot(fashion_mnist_data["loss"], label="Fashion MNIST")
    plt.title("Training Loss for Models with Hidden Size 64")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("plots", "training_loss_hidden_size_64.png"))
    plt.close()


def plot_mnist_fashion():
    """Plot training loss for Fashion MNIST with different hidden sizes."""
    with open("fashion_mnist_loss_hidden_size_64_iter_200.json", "r") as f:
        hidden_64_data = json.load(f)
    with open("fashion_mnist_loss_hidden_size_128_iter_200.json", "r") as f:
        hidden_128_data = json.load(f)

    plt.plot(hidden_64_data["loss"], label="Hidden Size 64")
    plt.plot(hidden_128_data["loss"], label="Hidden Size 128")
    plt.title("Training Loss for Fashion MNIST with Different Hidden Sizes")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(
        os.path.join("plots", "fashion_mnist_training_loss_different_hidden_sizes.png")
    )
    plt.close()


if __name__ == "__main__":
    plot_mnist_fashion()
