import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

    def get_recommendations(self, movie_id, top_n=5):
        """Zwroc rekomendacje podobnych filmow."""
        if self.labels is None:
            raise ValueError("Najpierw wykonaj klasteryzacje!")

        cluster = self.labels[movie_id]
        similar_indices = np.where(self.labels == cluster)[0]

        distances = []
        for idx in similar_indices:
            if idx != movie_id:
                dist = np.linalg.norm(self.features[movie_id] - self.features[idx])
                distances.append((idx, dist))

        distances.sort(key=lambda x: x[1])
        top = distances[:top_n]

        results = []
        for idx, dist in top:
            results.append(
                {
                    "title": self.data.iloc[idx]["title"],
                    "rating": self.data.iloc[idx]["rating"],
                    "similarity": round(1 / (1 + dist), 3),
                }
            )

        return results

    def show_clusters(self):
        """Wyswietl informacje o klastrach."""
        for i in range(self.num_clusters):
            cluster_data = self.data[self.labels == i]
            print(f"\nKlaster {i}:")
            print(f"  Filmy: {len(cluster_data)}")
            print(f"  Srednia ocena: {cluster_data['rating'].mean():.2f}")
            print(f"  Tytuly: {', '.join(cluster_data['title'].head(3).tolist())}")

    def visualize(self):
        """Wizualizuj klastry w 2D przy uzyciu PCA."""
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=self.labels,
            cmap="viridis",
            s=80,
            alpha=0.7,
        )

        for idx, title in enumerate(self.data["title"]):
            plt.annotate(
                title[:5],
                (features_2d[idx, 0], features_2d[idx, 1]),
                fontsize=7,
                alpha=0.8,
            )

        plt.colorbar(scatter, label="Klaster")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.title("Klasteryzacja filmow K-Means")
        plt.tight_layout()
        plt.savefig("clusters.png", dpi=150)
        print("Wykres zapisany: clusters.png")
        plt.close()

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

    def get_smart_recommendations(self, user_name):
        """Zwroc rekomendacje i antyrrekomendacje dla uzytkownika.

        Args:
            user_name (str): Imie i nazwisko uzytkownika

        Returns:
            dict: Slownik z rekomendacjami i antyrrekomendacjami
        """
        if user_name not in self.user_preferences:
            return {"error": "Uzytkownik nie znaleziony"}

        prefs = self.user_preferences[user_name]
        likes = prefs["likes"]
        dislikes = prefs["dislikes"]

        # Oblicz średnią ocenę filmów lubianych i nielubianych
        if likes:
            liked_avg = self.data.iloc[likes]["rating"].mean()
        else:
            liked_avg = 0

        if dislikes:
            disliked_avg = self.data.iloc[dislikes]["rating"].mean()
        else:
            disliked_avg = 10

        # Rekomenduj filmy podobne do lubianych (wysoka ocena)
        recommendations = []
        for idx in range(len(self.data)):
            if idx not in likes and idx not in dislikes:
                score = self.data.iloc[idx]["rating"]
                recommendations.append((idx, score))

        # Rekomenduj do unikania filmy podobne do nielubianych (niska ocena)
        avoid = []
        for idx in range(len(self.data)):
            if idx not in likes and idx not in dislikes:
                score = self.data.iloc[idx]["rating"]
                avoid.append((idx, score))

        # Sortuj - polecaj wysokie oceny, odradzaj niskie
        recommendations.sort(key=lambda x: abs(x[1] - liked_avg))
        avoid.sort(key=lambda x: abs(x[1] - disliked_avg))

        top_recommend = recommendations[:5]
        top_avoid = avoid[:5]

        return {
            "polecam": [
                {"tytul": self.data.iloc[idx]["title"], "ocena": rating}
                for idx, rating in top_recommend
            ],
            "odradzam": [
                {"tytul": self.data.iloc[idx]["title"], "ocena": rating}
                for idx, rating in top_avoid
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

    args = parser.parse_args()

    print("=" * 60)
    print("SYSTEM REKOMENDACJI FILMOW")
    print("=" * 60)

    recommender = FilmRecommender(num_clusters=3)
    recommender.load_data()
    recommender.prepare_features()
    recommender.cluster()

    print("\n--- Statystyki klastrów ---")
    recommender.show_clusters()

    print("\n--- Rekomendacje dla 'Shawshank Redemption' ---")
    recs = recommender.get_recommendations(movie_id=0, top_n=5)
    for i, rec in enumerate(recs, 1):
        print(
            f"{i}. {rec['title']} (ocena: {rec['rating']}, podobienstwo: {rec['similarity']})"
        )

    print("\n--- Wizualizacja ---")
    recommender.visualize()

    # osobiste rekomendacje dla wybranego uzytkownika
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

        print(f"\n--- Polecam dla {user_name} ---")
        for i, film in enumerate(result["polecam"], 1):
            print(f"{i}. {film['tytul']} (ocena: {film['ocena']})")

        print(f"\n--- Odradzam dla {user_name} ---")
        for i, film in enumerate(result["odradzam"], 1):
            print(f"{i}. {film['tytul']} (ocena: {film['ocena']})")
    else:
        print("DOSTEPNI UZYTKOWNICY:")
        print("=" * 60)
        for i, user in enumerate(recommender.user_data.keys(), 1):
            print(f"{i}. {user}")
        print("\nUzyj: python main.py --user 'Imie Nazwisko'")


if __name__ == "__main__":
    main()
