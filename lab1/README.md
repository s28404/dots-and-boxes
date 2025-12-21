# Dots and Boxes

Krótki opis
-----------
To repozytorium zawiera prostą implementację gry "Dots and Boxes" w Pythonie. Gra pozwala na rozgrywkę pomiędzy AI (Player 1) korzystającym z algorytmu alpha-beta oraz graczem ludzkim (Player 2) w trybie tekstowym.

Zasady gry
----------
Zasady gry (po angielsku) są dostępne tutaj:

https://en.wikipedia.org/wiki/Dots_and_Boxes

Autorzy
-------
Kajetan Frąckowiak

Przygotowanie środowiska
------------------------
1. Upewnij się, że masz zainstalowanego Pythona 3.8+.
2. (Opcjonalnie) Utwórz wirtualne środowisko:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Nie ma dodatkowych zależności zewnętrznych — wystarczy uruchomić skrypt.

Uruchamianie gry
-----------------
W katalogu projektu uruchom:

```bash
python main.py
```

Instrukcja obsługi
------------------
W trakcie tury gracza wpisz ruch w formacie:

```
[h|v] R C
```

gdzie `h` oznacza linię poziomą (horizontal) a `v` linię pionową (vertical). `R` i `C` to indeksy wiersza/kolumny (od 0).

Przykład: `h 0 0` rysuje górną, poziomą linię zaczynającą się w punkcie (0,0).

Uwagi
-----
Jeśli chcesz zmienić rozmiar planszy lub parametry AI, edytuj stałe w `main.py`.
