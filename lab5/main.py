#############################
#   Autorzy:
#   Kajetan Frąckowiak s28404
#   Marek Walkowski    s25378
#############################
"""
PROBLEM:
Implementacja i trening sieci neuronowej do klasyfikacji danych z różnych zbiorów danych
(Iris, CIFAR-10, Fashion MNIST, Heart Disease UCI).

OPIS:
Program wykorzystuje sklearn.neural_network.MLPClassifier do budowy i trenowania
wielowarstwowego perceptronu. Obsługuje różne rozmiary warstwy ukrytej oraz
zmienną liczbę iteracji treningowych. Program zapisuje historię straty treningowej
do pliku JSON i oblicza metryki klasyfikacji.

INSTRUKCJA UŻYCIA:
1. Instalacja zależności:
   pip install -r requirements.txt

2. Trening modelu:
   python main.py --dataset <dataset> --hidden_size <size> --max_iter <iter>

3. Przykłady:
   # Domyślnie (Heart Disease, hidden_size=64, max_iter=200)
   python main.py

   # Fashion MNIST z warstwą 128
   python main.py --dataset fashion_mnist --hidden_size 128

   # CIFAR-10
   python main.py --dataset cifar10

REFERENCJE:
- https://scikit-learn.org/stable/modules/neural_networks.html
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse
import json


class SimpleNN:
    """Minimal neural network wrapper around sklearn's MLPClassifier."""

    def __init__(self, hidden_size=64, max_iter=200, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_size,),
            max_iter=max_iter,
            random_state=random_state,
            verbose=1,
            early_stopping=False,  # Wyłącz early stopping
            n_iter_no_change=1000,  # Zwiększ cierpliwość
        )

    def fit(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def score(self, X, y):
        """Calculate accuracy score."""
        return self.model.score(X, y)


class Trainer:
    """Minimal trainer for neural networks."""

    def __init__(self, model=None):
        self.model = model if model else SimpleNN()
        self.training_loss = []

    def train(self, X_train, y_train):
        """Train the model and capture loss."""
        print("Training...")
        self.model.fit(X_train, y_train)
        # Capture loss history from sklearn's loss_curve_
        if hasattr(self.model.model, "loss_curve_"):
            loss = self.model.model.loss_curve_
            self.training_loss = loss.tolist() if hasattr(loss, "tolist") else loss

    def save_loss_to_json(self, filename="training_loss.json"):
        """Save training loss to JSON file."""
        # Convert numpy types to Python native types
        loss_list = [float(x) for x in self.training_loss]
        data = {"loss": loss_list, "iterations": len(loss_list)}
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Training loss saved to {filename}")

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return predictions."""
        preds = self.model.predict(X_test)
        return preds, y_test

    def get_metrics(self, preds, labels):
        """Get confusion matrix and classification report."""
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds)
        accuracy = np.mean(preds == labels)

        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "accuracy": accuracy,
        }


if __name__ == "__main__":
    from data_loaders import (
        get_heart_disease_data,
        get_iris_data,
        get_cifar10_data,
        get_fashion_mnist_data,
    )

    parser = argparse.ArgumentParser(
        description="Train neural network on different datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["disease", "iris", "cifar10", "fashion_mnist"],
        default="disease",
        help="Dataset to train on (default: disease)",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="Hidden layer size (default: 64)"
    )
    parser.add_argument(
        "--max_iter", type=int, default=200, help="Maximum iterations (default: 200)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "disease":
        X_train, X_test, y_train, y_test = get_heart_disease_data()
    elif args.dataset == "iris":
        X_train, X_test, y_train, y_test = get_iris_data()
    elif args.dataset == "cifar10":
        X_train, X_test, y_train, y_test = get_cifar10_data()
    elif args.dataset == "fashion_mnist":
        X_train, X_test, y_train, y_test = get_fashion_mnist_data()

    print("Creating model...")
    model = SimpleNN(hidden_size=args.hidden_size, max_iter=args.max_iter)

    print("Training...")
    trainer = Trainer(model)
    trainer.train(X_train, y_train)

    # Save training loss to JSON
    trainer.save_loss_to_json(
        f"{args.dataset}_loss_hidden_size_{args.hidden_size}_iter_{args.max_iter}.json"
    )

    print("\nEvaluating...")
    preds, labels = trainer.evaluate(X_test, y_test)
    metrics = trainer.get_metrics(preds, labels)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["classification_report"])
