# System Rekomendacji Filmów

System rekomendacji filmów i seriali wykorzystujący klasteryzację K-Means oraz TMDb API.

## Opis

System:
1. Grupuje 20 filmów/seriali na podstawie cech (ocena, czas trwania, liczba głosów, typ)
2. Analizuje preferencje użytkowników
3. **Wyszukuje nowe filmy przez TMDb API** dopasowane do gustu użytkownika
4. Udziela:
   - **Rekomendacji** - nowe filmy z TMDb z lubianych gatunków
   - **Antyrrekomendacji** - nowe filmy z nielubianych gatunków

## Instalacja

```bash
pip install -r requirements.txt
```

## Użytkowanie

### Podstawowe użycie
```bash
python main.py --user "Kajetan Frackowiak"
```

### Z dodatkowymi informacjami o filmach
```bash
python main.py --user "Kajetan Frackowiak" --info
```

### Lista dostępnych użytkowników
```bash
python main.py
```

## Dostępni użytkownicy

- Pawel Czapiewski
- Kacper Olejnik
- Pawel Kleszyk
- Kajetan Frackowiak
- Michal Fritza
- Jan Skulimowski
- Kamil Littwitz
- Stefan Karczewski
- Wiktor Swierzynski
- Kacper Pach
- Marek Walkowski

## API - Główne metody

### `FilmRecommender(num_clusters=3)`
Inicjalizuje silnik rekomendacji z K-Means.

### `load_data()`
Załadowuje bazę 20 filmów/seriali do klasteryzacji.

### `prepare_features()`
Normalizuje cechy filmów (StandardScaler).

### `cluster()`
Wykonuje klasteryzację K-Means na filmach.

### `add_user_opinion(user_name, likes_idx, dislikes_idx)`
Zapisuje preferencje użytkownika.
- `user_name`: Imię i nazwisko
- `likes_idx`: Indeksy filmów które lubi
- `dislikes_idx`: Indeksy filmów których nie lubi

### `search_movies_by_genre(genre_ids, exclude_genre_ids, limit)`
Wyszukuje filmy z TMDb API na podstawie gatunków.
- `genre_ids`: Lista ID gatunków do wyszukania
- `exclude_genre_ids`: Lista ID gatunków do wykluczenia
- `limit`: Liczba filmów do zwrócenia

### `get_smart_recommendations(user_name)`
Zwraca rekomendacje i antyrrekomendacje z TMDb API.
- **Polecam**: 5 nowych filmów z lubianych gatunków
- **Odradzam**: 5 nowych filmów z nielubianych gatunków

## Zewnętrzne API

System używa **TMDb API** (The Movie Database):
- Adres: https://api.themoviedb.org/3/
- Darmowy klucz API zawarty w kodzie
- Zwraca popularne filmy z określonych gatunków

### Przykład dla Kajetana Frąckowiaka:
- **Lubi:** Star Wars, Harry Potter, Shrek, Avengers (10/10)
- **Nie lubi:** 50 twarzy Greya, 365 dni (2-3/10)
- **System poleca:** Action, Adventure, Fantasy, Sci-Fi, Animation
- **System odradza:** Romance, Drama (filmy erotyczne)

## Algorytm

1. **K-Means clustering** grupuje 20 filmów bazowych
2. **Analiza preferencji** określa lubiane/nielubiane gatunki
3. **TMDb API** wyszukuje nowe filmy spoza bazy
4. **Rekomendacje** dopasowane do gustu użytkownika

### Cechy użyte do klasteryzacji:
- `rating` - ocena IMDB (0-10)
- `duration` - czas trwania (minuty)
- `votes` - liczba głosów IMDB
- `type` - typ produkcji (0=film, 1=serial)

## Złożoność

K-Means: **O(nkl)**
- n = liczba filmów (20)
- k = liczba klastrów (3)
- l = liczba iteracji (~10)

## Pliki

- `main.py` - główny skrypt (379 linii)
- `requirements.txt` - zależności Python
- `README.md` - dokumentacja

