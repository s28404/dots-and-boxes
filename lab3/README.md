# System Rekomendacji Filmów

Prosty system rekomendacji filmów i seriali oparty na klasteryzacji K-Means.

## Opis

System grupuje filmy/seriale na podstawie ich charakterystyk (ocena IMDB, czas trwania, liczba głosów, typ) i udziela:
- **Rekomendacji** - podobne filmy z tego samego klastra
- **Antyrrekomendacji** - filmy do unikania na podstawie preferencji użytkownika

## Instalacja

```bash
pip install -r requirements.txt
```

## Użytkowanie

```bash
python main.py
```

Program:
1. Załaduje 20 filmów/seriali
2. Znormalizuje cechy
3. Zastosuje K-Means (3 klastry)
4. Wyświetli statystyki klastrów
5. Pokaże rekomendacje dla "Shawshank Redemption"
6. Wygeneruje wizualizację (clusters.png)
7. **Pokaże osobiste rekomendacje dla użytkownika** (Jan Kowalski)

## API - Główne funkcje

### `FilmRecommender(num_clusters=3)`
Inicjalizuje silnik rekomendacji.

### `load_data()`
Załadowuje przykładowe dane o filmach/serialach.

### `prepare_features()`
Normalizuje cechy do klasteryzacji.

### `cluster()`
Wykonuje klasteryzację K-Means.

### `add_user_opinion(user_name, likes_idx, dislikes_idx)`
Zapisuje opinie użytkownika.
- `user_name`: Imię i nazwisko
- `likes_idx`: Indeksy filmów które się podoba
- `dislikes_idx`: Indeksy filmów które się nie podoba

### `get_smart_recommendations(user_name)`
Zwraca rekomendacje i antyrrekomendacje dla użytkownika.
- **Polecam**: 5 filmów z lubanych klastrów
- **Odradzam**: 5 filmów z nielubianych klastrów

## Pliki

- `main.py` - główny skrypt z całą logika
- `requirements.txt` - zależności
- `clusters.png` - wizualizacja klastrów (generowana przy uruchomieniu)

## Algorytm

Używamy K-Means do podziału filmów na grupy. Rekomendacje są dobierane z lubanych klastrów, antyrrekomendacje z nielubianych.

### Cechy użyte:
- rating - ocena IMDB (0-10)
- duration - czas trwania (minuty)
- votes - liczba głosów IMDB
- type - typ (0=film, 1=serial)

## Złożoność

K-Means: O(nkl) gdzie:
- n = liczba filmów (20)
- k = liczba klastrów (3)
- l = liczba iteracji (~10)

