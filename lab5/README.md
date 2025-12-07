# Neural Network Classifier Project

Projekt implementujący prosty klasyfikator neuronowy przy użyciu scikit-learn.

## Instalacja zależności

```bash
pip install -r requirements.txt
```

## Uruchomienie projektu

Aby wytrenować model:

```bash
python main.py --dataset <dataset> --hidden_size <size> --max_iter <iter>
```

### Dostępne parametry:

- `--dataset`: Wybór zbioru danych (opcje: `iris`, `fashion_mnist`, `disease`, `cifar10`)
  - `iris` - Iris dataset
  - `fashion_mnist` - Fashion MNIST dataset
  - `disease` - Heart Disease dataset
  - `cifar10` - CIFAR-10 dataset
- `--hidden_size`: Rozmiar warstwy ukrytej (domyślnie: 64)
- `--max_iter`: Maksymalna liczba iteracji (domyślnie: 200)

### Przykłady:

```bash
# Iris dataset z warstwą ukrytą rozmiaru 64
python main.py --dataset iris

# Fashion MNIST z warstwą ukrytą rozmiaru 128 i 200 iteracjami
python main.py --dataset fashion_mnist --hidden_size 128 --max_iter 200

# Heart Disease dataset
python main.py --dataset disease --hidden_size 64
```

## Wyniki treningowe

### Fashion MNIST - Hidden Size 64
![Fashion MNIST Overview Hidden 64](fashion_mnist_hidden_64.png)

### Fashion MNIST - Hidden Size 128
![Fashion MNIST Overview Hidden 128](fashion_mnist_hidden_128.png)

### Learning Curve - All datasets with Hidden Size 64
![Learning Curve All Datasets](plots/training_loss_hidden_size_64.png)

### Learning Curve - Fashion MNIST with different Hidden Sizes
![Learning Curve Different Hidden Sizes](plots/fashion_mnist_training_loss_different_hidden_sizes.png)

## Autorzy

- Kajetan Frąckowiak (s28404)
- Marek Walkowski (s25378)

