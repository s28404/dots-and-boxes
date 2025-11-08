"""
Przyklad uzycia API systemu rekomendacji
"""

from main import FilmRecommender


def example_api_usage():
    """Przyklad jak uzywac API dla rownlych uzytkownikow."""

    # Inicjalizacja
    rec = FilmRecommender(num_clusters=3)
    rec.load_data()
    rec.prepare_features()
    rec.cluster()

    # Uzytkownik 1: Maria Lewandowska
    # lubi: Shawshank(0), Godfather(1), Inception(5)
    # nie lubi: Breaking Bad(10), Sopranos(11)
    rec.add_user_opinion(
        user_name="Maria Lewandowska", likes_idx=[0, 1, 5], dislikes_idx=[10, 11]
    )

    # Uzytkownik 2: Piotr Nowak
    # lubi: Dark Knight(2), Matrix(7), Interstellar(8)
    # nie lubi: Game of Thrones(12), Peaky Blinders(13)
    rec.add_user_opinion(
        user_name="Piotr Nowak", likes_idx=[2, 7, 8], dislikes_idx=[12, 13]
    )

    # Pobierz rekomendacje dla Maria
    print("=" * 50)
    print("REKOMENDACJE DLA MARII LEWANDOWSKIEJ")
    print("=" * 50)
    result = rec.get_smart_recommendations("Maria Lewandowska")

    print("\nPolecam do oglądania:")
    for i, film in enumerate(result["polecam"], 1):
        print(f"  {i}. {film['tytul']} (ocena: {film['ocena']})")

    print("\nOdradzam do oglądania:")
    for i, film in enumerate(result["odradzam"], 1):
        print(f"  {i}. {film['tytul']} (ocena: {film['ocena']})")

    # Pobierz rekomendacje dla Piotra
    print("\n" + "=" * 50)
    print("REKOMENDACJE DLA PIOTRA NOWAKA")
    print("=" * 50)
    result = rec.get_smart_recommendations("Piotr Nowak")

    print("\nPolecam do oglądania:")
    for i, film in enumerate(result["polecam"], 1):
        print(f"  {i}. {film['tytul']} (ocena: {film['ocena']})")

    print("\nOdradzam do oglądania:")
    for i, film in enumerate(result["odradzam"], 1):
        print(f"  {i}. {film['tytul']} (ocena: {film['ocena']})")


if __name__ == "__main__":
    example_api_usage()
