#############################
#   Autorzy:
#   Kajetan Frąckowiak s28404
#   Marek Walkowski    s25378
#############################
"""
PROBLEM:
Załadowanie i przygotowanie zbiorów danych do trenowania sieci neuronowej
w spójny i powtarzalny sposób.

OPIS:
Moduł zawiera funkcje do ładowania czterech zbiorów danych:
- Iris (lokalnie z sklearn)
- Heart Disease/Breast Cancer UCI (lokalnie z sklearn)
- Fashion MNIST (z Hugging Face)
- CIFAR-10 (z Hugging Face)

Każda funkcja normalizuje dane za pomocą StandardScaler i dzieli je na
zbiór treningowy (80%) i testowy (20%).

INSTRUKCJA UŻYCIA:
from data_loaders import get_fashion_mnist_data, get_iris_data
X_train, X_test, y_train, y_test = get_fashion_mnist_data()

REFERENCJE:
- https://scikit-learn.org/stable/datasets/toy_dataset.html
- https://huggingface.co/datasets/fashion_mnist
- https://huggingface.co/datasets/cifar10
"""

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np


# ==================== Heart Disease UCI Dataset ====================
def get_heart_disease_data(test_size=0.2):
    """
    Load Heart Disease (Breast Cancer) dataset from sklearn.
    Returns X_train, X_test, y_train, y_test
    """
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"Heart Disease: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features, 2 classes"
    )

    return X_train, X_test, y_train, y_test


# ==================== Fisher Iris Dataset ====================
def get_iris_data(test_size=0.2):
    """
    Load Fisher Iris dataset from sklearn.
    Returns X_train, X_test, y_train, y_test
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"Iris: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features, 3 classes"
    )

    return X_train, X_test, y_train, y_test


# ==================== CIFAR-10 Dataset ====================
def get_cifar10_data():
    """
    Load CIFAR-10 dataset from Hugging Face.
    Returns X_train, X_test, y_train, y_test (flattened)
    """
    dataset = load_dataset("cifar10")

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for sample in dataset["train"]:
        img = np.array(sample["img"], dtype=np.float32).flatten()
        X_train.append(img)
        y_train.append(sample["label"])

    for sample in dataset["test"]:
        img = np.array(sample["img"], dtype=np.float32).flatten()
        X_test.append(img)
        y_test.append(sample["label"])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"CIFAR-10: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features, 10 classes"
    )

    return X_train, X_test, y_train, y_test


# ==================== Fashion MNIST Dataset ====================
def get_fashion_mnist_data():
    """
    Load Fashion MNIST dataset from Hugging Face.
    Returns X_train, X_test, y_train, y_test (flattened)
    """
    dataset = load_dataset("fashion_mnist")

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for sample in dataset["train"]:
        img = np.array(sample["image"], dtype=np.float32).flatten()
        X_train.append(img)
        y_train.append(sample["label"])

    for sample in dataset["test"]:
        img = np.array(sample["image"], dtype=np.float32).flatten()
        X_test.append(img)
        y_test.append(sample["label"])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"Fashion MNIST: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features, 10 classes"
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Loading all datasets...\n")
    get_heart_disease_data()
    get_iris_data()
    get_cifar10_data()  # Uncomment to load (large download)
    get_fashion_mnist_data()  # Uncomment to load (large download)
