# Klasyfikacja: Drzewa Decyzyjne i SVM

Projekt zawiera rozwiązania zadań laboratoryjnych dotyczących klasyfikacji danych przy użyciu algorytmów uczenia maszynowego: Drzew Decyzyjnych (Decision Trees) oraz Maszyny Wektorów Nośnych (SVM).

## Zawartość

W katalogu znajdują się dwa główne skrypty realizujące zadanie dla dwóch różnych zbiorów danych:

1.  **`classification_diabetes.py`**
    *   **Zbiór danych:** Pima Indians Diabetes Dataset.
    *   **Cel:** Przewidywanie wystąpienia cukrzycy na podstawie parametrów medycznych.
    *   **Funkcje:**
        *   Pobieranie danych z repozytorium online.
        *   Wizualizacja danych 2D przy użyciu PCA.
        *   Trening i ewaluacja klasyfikatorów (Drzewo Decyzyjne, SVM).
        *   Porównanie skuteczności różnych funkcji jądra (kernels) dla SVM.

2.  **`classification_heart_disease.py`**
    *   **Zbiór danych:** Heart Disease UCI (Cleveland).
    *   **Cel:** Przewidywanie obecności choroby serca.
    *   **Funkcje:** Analogiczne do pierwszego skryptu (wizualizacja, klasyfikacja, porównanie kerneli).

## Wymagania

Aby uruchomić projekt, potrzebujesz zainstalowanego Pythona oraz bibliotek wymienionych w pliku `requirements.txt`.

### Instalacja zależności

Uruchom poniższą komendę w terminalu:

```bash
pip install -r requirements.txt
```

## Uruchomienie

Aby uruchomić analizę dla zbioru Diabetes:

```bash
python classification_diabetes.py
```

Aby uruchomić analizę dla zbioru Heart Disease:

```bash
python classification_heart_disease.py
```

## Wyniki

Każdy skrypt po uruchomieniu:
1.  Wyświetli podgląd danych.
2.  Wygeneruje wykres (okno z wykresem należy zamknąć, aby skrypt kontynuował działanie).
3.  Wypisze w terminalu metryki klasyfikacji (Accuracy, Precision, Recall, F1-Score).
4.  Pokaże wynik predykcji dla przykładowego pacjenta.
5.  Przedstawi porównanie wyników dla różnych parametrów SVM (linear, poly, rbf, sigmoid).
