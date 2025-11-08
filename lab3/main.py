import pandas as pd
import argparse
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class FilmRecommender:
    """Silnik rekomendacji filmow/seriali oparty na klasteryzacji K-Means.

    Grupuje produkcje na podstawie oceny, czasu trwania, liczby glosow i typu.
    Udziela rekomendacji i antyrrekomendacji na podstawie preferencji uzytkownika.
    """

    def __init__(self, num_clusters=3):
        """Inicjalizacja rekomendatora.

        Args:
            num_clusters (int): Liczba klastrów K-Means. Default: 3
        """
        self.num_clusters = num_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.data = None
        self.features = None
        self.labels = None
        self.user_preferences = {}
        self.user_data = self._init_user_data()

    def _init_user_data(self):
        """Inicjalizuj dane uzytkownikow z ich preferencjami filmow."""
        return {
            "Pawel Czapiewski": {
                "likes": [5, 8, 9, 17, 18, 19],
                "dislikes": [0, 10, 11, 12, 13, 14],
            },
            "Kacper Olejnik": {"likes": [0, 1, 2, 6, 7, 15], "dislikes": [3, 10, 11]},
            "Pawel Kleszyk": {
                "likes": [0, 1, 2, 6, 9, 14, 16, 18],
                "dislikes": [4, 5, 10, 11, 13, 15],
            },
            "Kajetan Frackowiak": {
                "likes": [0, 2, 5, 6, 7, 8, 9],
                "dislikes": [10, 11, 12, 13, 14, 15],
            },
            "Michal Fritza": {"likes": [1, 5, 7, 8, 9], "dislikes": [10, 11, 12, 13]},
            "Jan Skulimowski": {"likes": [0, 1, 2, 5, 9], "dislikes": [10, 14, 15, 16]},
            "Kamil Littwitz": {
                "likes": [5, 8, 9, 14, 16, 18, 19],
                "dislikes": [0, 10, 11, 12, 13],
            },
            "Stefan Karczewski": {
                "likes": [0, 1, 2, 5, 8, 9],
                "dislikes": [10, 12, 13, 15],
            },
            "Wiktor Swierzynski": {"likes": [1, 2, 5, 8], "dislikes": [10, 12, 14]},
            "Kacper Pach": {"likes": [0, 1, 2, 5, 9], "dislikes": [10, 11, 12, 13]},
            "Marek Walkowski": {"likes": [2, 5, 7, 9], "dislikes": [10, 14, 15]},
        }

    def load_data(self):
        """Zaladuj przykladowe dane o filmach i serialach."""
        movies = {
            "title": [
                "Shawshank Redemption",
                "Godfather",
                "Dark Knight",
                "Pulp Fiction",
                "Forrest Gump",
                "Inception",
                "Fight Club",
                "Matrix",
                "Interstellar",
                "Gladiator",
                "Breaking Bad",
                "Sopranos",
                "Game of Thrones",
                "Peaky Blinders",
                "Stranger Things",
                "Chernobyl",
                "The Crown",
                "True Detective",
                "Westworld",
                "Mr Robot",
            ],
            "rating": [
                9.3,
                9.2,
                9.0,
                8.9,
                8.8,
                8.8,
                8.8,
                8.7,
                8.6,
                8.5,
                9.5,
                9.2,
                9.3,
                8.6,
                8.7,
                9.3,
                8.6,
                8.8,
                8.5,
                8.4,
            ],
            "duration": [
                142,
                175,
                152,
                154,
                142,
                148,
                139,
                136,
                169,
                155,
                58,
                52,
                56,
                60,
                50,
                45,
                58,
                61,
                65,
                50,
            ],
            "votes": [
                2300000,
                1700000,
                2800000,
                1900000,
                1800000,
                2400000,
                1900000,
                2000000,
                1500000,
                1400000,
                300000,
                250000,
                320000,
                150000,
                180000,
                100000,
                120000,
                140000,
                160000,
                170000,
            ],
            "type": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        self.data = pd.DataFrame(movies)
        print(f"Zaladowano {len(self.data)} filmow/seriali")

    def prepare_features(self):
        """Przygotuj i znormalizuj cechy do klasteryzacji."""
        features_list = ["rating", "duration", "votes", "type"]
        X = self.data[features_list].values
        self.features = self.scaler.fit_transform(X)
        print(f"Przygotowano cechy: {self.features.shape}")

    def cluster(self):
        """Wykonaj klasteryzacje K-Means."""
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.features)
        print(f"Klasteryzacja: {self.num_clusters} klastery")

    def add_user_opinion(self, user_name, likes_idx, dislikes_idx):
        """Zapisz opinie uzytkownika o filmach.

        Args:
            user_name (str): Imie i nazwisko uzytkownika
            likes_idx (list): Indeksy filmow ktorych uzytkownik lubi
            dislikes_idx (list): Indeksy filmow ktorych nie lubi
        """
        self.user_preferences[user_name] = {
            "likes": likes_idx,
            "dislikes": dislikes_idx,
        }
        print(f"Zapisano opinie dla {user_name}")

    def search_movies_by_genre(self, genre_ids, exclude_genre_ids=None, limit=5):
        """Wyszukaj filmy z TMDb API na podstawie gatunku.

        Args:
            genre_ids (list): Lista ID gatunków do wyszukania
            exclude_genre_ids (list): Lista ID gatunków do wykluczenia
            limit (int): Liczba filmów do zwrócenia

        Returns:
            list: Lista słowników z informacjami o filmach
        """
        api_key = "8265bd1679663a7ea12ac168da84d2e8"

        # Pobierz popularne filmy z danego gatunku
        genre_str = ",".join(map(str, genre_ids))
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&language=pl&sort_by=popularity.desc&with_genres={genre_str}&vote_count.gte=100"

        if exclude_genre_ids:
            exclude_str = ",".join(map(str, exclude_genre_ids))
            url += f"&without_genres={exclude_str}"

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = []

                for movie in data.get("results", [])[:limit]:
                    results.append(
                        {
                            "tytul": movie.get("title", "N/A"),
                            "rok": movie.get("release_date", "N/A")[:4],
                            "ocena_tmdb": movie.get("vote_average", "N/A"),
                            "opis": movie.get("overview", "Brak opisu"),
                        }
                    )

                return results
        except Exception as e:
            print(f"Blad wyszukiwania filmów: {e}")

        return []

    def get_smart_recommendations(self, user_name):
        """Zwroc rekomendacje i antyrrekomendacje dla uzytkownika.

        Rekomendacje pochodzą z TMDb API (nowe filmy) na podstawie preferencji.

        Args:
            user_name (str): Imie i nazwisko uzytkownika

        Returns:
            dict: Slownik z rekomendacjami i antyrrekomendacjami z API
        """
        if user_name not in self.user_preferences:
            return {"error": "Uzytkownik nie znaleziony"}

        # Gatunki TMDb:
        # 28=Action, 12=Adventure, 16=Animation, 35=Comedy, 80=Crime,
        # 18=Drama, 14=Fantasy, 27=Horror, 9648=Mystery, 10749=Romance,
        # 878=Science Fiction, 53=Thriller, 10752=War

        # Określ gatunki na podstawie użytkownika
        genre_mapping = {
            "Kajetan Frackowiak": {
                "likes": [
                    28,
                    12,
                    14,
                    878,
                    16,
                ],  # Action, Adventure, Fantasy, Sci-Fi, Animation (Star Wars, Harry Potter, Shrek, Avengers)
                "dislikes": [10749, 18],  # Romance, Drama (50 twarzy greya, 365 dni)
            },
            # Inne użytkownicy domyślne gatunki
            "default": {
                "likes": [28, 80, 18],  # Action, Crime, Drama
                "dislikes": [27, 10749],  # Horror, Romance
            },
        }

        user_genres = genre_mapping.get(user_name, genre_mapping["default"])

        # Wyszukaj nowe filmy do polecenia (z lubianych gatunków, bez nielubianych)
        recommendations = self.search_movies_by_genre(
            genre_ids=user_genres["likes"],
            exclude_genre_ids=user_genres["dislikes"],
            limit=5,
        )

        # Wyszukaj filmy do odradzania (z nielubianych gatunków)
        avoid_list = self.search_movies_by_genre(
            genre_ids=user_genres["dislikes"],
            exclude_genre_ids=user_genres["likes"],
            limit=5,
        )

        return {
            "polecam": [
                {
                    "tytul": film["tytul"],
                    "ocena": film["ocena_tmdb"],
                    "rok": film["rok"],
                    "opis": film["opis"],
                }
                for film in recommendations
            ],
            "odradzam": [
                {
                    "tytul": film["tytul"],
                    "ocena": film["ocena_tmdb"],
                    "rok": film["rok"],
                    "opis": film["opis"],
                }
                for film in avoid_list
            ],
        }


def main():
    parser = argparse.ArgumentParser(
        description="System rekomendacji filmow - podaj imie i nazwisko uzytkownika"
    )
    parser.add_argument(
        "--user",
        "-u",
        type=str,
        help="Imie i nazwisko uzytkownika (np. 'Pawel Czapiewski')",
    )
    parser.add_argument(
        "--info",
        "-i",
        action="store_true",
        help="Pobierz dodatkowe informacje o filmach z API",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SYSTEM REKOMENDACJI FILMOW")
    print("=" * 60)

    recommender = FilmRecommender(num_clusters=3)
    recommender.load_data()
    recommender.prepare_features()
    recommender.cluster()

    print(f"Zaladowano {len(recommender.data)} filmow/seriali")
    print(f"Klasteryzacja: {recommender.num_clusters} klastery")

    # Osobiste rekomendacje dla wybranego uzytkownika
    print("\n" + "=" * 60)

    if args.user:
        user_name = args.user
        if user_name not in recommender.user_data:
            print(f"Blad: '{user_name}' nie znaleziony w bazie.")
            print("\nDostepni uzytkownicy:")
            for u in recommender.user_data.keys():
                print(f"  - {u}")
            return

        # Załaduj opinie użytkownika z bazy
        user_prefs = recommender.user_data[user_name]
        recommender.add_user_opinion(
            user_name=user_name,
            likes_idx=user_prefs["likes"],
            dislikes_idx=user_prefs["dislikes"],
        )

        result = recommender.get_smart_recommendations(user_name)

        print(f"OSOBISTE REKOMENDACJE DLA {user_name.upper()}")
        print("=" * 60)

        print(f"\n--- Polecam dla {user_name} (nowe filmy z TMDb) ---")
        for i, film in enumerate(result["polecam"], 1):
            print(f"{i}. {film['tytul']} ({film['rok']}) - ocena TMDb: {film['ocena']}")
            if args.info and film.get("opis"):
                print(f"   Opis: {film['opis'][:150]}...")

        print(f"\n--- Odradzam dla {user_name} (nowe filmy z TMDb) ---")
        for i, film in enumerate(result["odradzam"], 1):
            print(f"{i}. {film['tytul']} ({film['rok']}) - ocena TMDb: {film['ocena']}")
            if args.info and film.get("opis"):
                print(f"   Opis: {film['opis'][:150]}...")
    else:
        print("DOSTEPNI UZYTKOWNICY:")
        print("=" * 60)
        for i, user in enumerate(recommender.user_data.keys(), 1):
            print(f"{i}. {user}")
        print("\nUzyj: python main.py --user 'Imie Nazwisko'")


if __name__ == "__main__":
    main()
