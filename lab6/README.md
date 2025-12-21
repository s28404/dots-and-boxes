# Detekcja Gestów - MediaPipe

Prosty program do rozpoznawania gestów rąk w czasie rzeczywistym za pomocą kamery.

## Wymagania

```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python gesture_detector.py
```

## Obsługiwane gesty

| Gest | Opis |
|------|------|
| **OPEN_HAND** | Ręka całkowicie otwarta - wszystkie palce wyprostowane |
| **FIST** | Pięść - wszystkie palce zaciśnięte |
| **THUMBS_UP** | Kciuk do góry, reszta zaciśnięta |
| **PEACE** | Dwa palce do góry (indeks + środkowy) |
| **OK_SIGN** | Kółeczko z kciuka i indeksu, reszta wyprostowana |

## Jak to działa

1. **Kamera** - `OpenCV` przechwytuje obraz
2. **MediaPipe Hands** - Wykrywa pozycję dłoni (21 punktów na rękę)
3. **Analiza landmarków** - Sprawdzamy pozycję każdego palca
4. **Klasyfikacja** - Na podstawie pozycji palców rozpoznajemy gest
5. **Wyświetlenie** - Rysujemy dłoń i pokazujemy rozpoznany gest

## Technika

- **Landmarki** - 21 punktów na dłoni (kostki palców, przegub)
- **Y-oś** - Używamy do sprawdzenia czy palec jest wyprostowany czy zaciśnięty
- **Historia gestów** - Przechowujemy ostatnie 5 gestów dla stabilności

## Sterowanie

- `Q` - Wyjście z programu
- Kamera włącza się automatycznie

## Optymalizacja

Jeśli program nie wykrywa dobrze:
- Lepsze oświetlenie
- Mniejsza odległość od kamery
- Spróbuj powiększyć `min_detection_confidence` w `gesture_detector.py` (może być wolniej, ale dokładniej)

## Rozwój

Możesz łatwo dodać nowe gesty dodając nową funkcję w `gesture_detector.py`:

```python
def is_rock(landmarks):
    # Twoja logika
    return condition
```

I dodaj do słownika `gestures` w funkcji `detect_gesture()`.
