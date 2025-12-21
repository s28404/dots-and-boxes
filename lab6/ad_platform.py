#############################
#   Autorzy:
#   Kajetan Frąckowiak s28404
#   Marek Walkowski    s25378
#############################
"""Platforma reklamowa wykorzystująca detekcję oczu do zarządzania odtwarzaniem reklam."""
import cv2
import time


class AdPlatform:
    """Klasa platformy reklamowej z detekcją oczu.

    Wykorzystuje algorytmy Haar Cascade z OpenCV do wykrywania twarzy i oczu.
    Automatycznie pauzuje reklamę gdy użytkownik nie patrzy.

    Attributes:
        face_cascade: Klasyfikator Haar Cascade do detekcji twarzy
        eye_cascade: Klasyfikator Haar Cascade do detekcji oczu
        is_watching: Bool określający czy użytkownik patrzy
        not_watching_time: Czas w sekundach od ostatniego wykrycia oczu
        ad_paused: Bool określający czy reklama jest zapauzowana
        ad_watched_time: Faktyczny czas obejrzenia reklamy w sekundach
        ad_duration: Całkowity czas trwania reklamy (30s)
    """

    def __init__(self):
        """Inicjalizacja platformy reklamowej.

        Wczytuje klasyfikatory Haar Cascade dla twarzy i oczu.
        Inicjalizuje zmienne stanu do śledzenia oglądania reklamy.
        """
        # Klasyfikatory do detekcji twarzy i oczu
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self.is_watching = False
        self.not_watching_time = 0
        self.last_time = time.time()
        self.ad_paused = False
        self.ad_watched_time = 0
        self.ad_duration = 30

    def detect_eyes(self, frame):
        """FUNKCJA 1: Detekcja oczu - sprawdza czy oglądający nie zamknął oczu.

        Wykrywa twarz za pomocą Haar Cascade, następnie w obszarze twarzy
        szuka oczu. Wymaga wykrycia co najmniej 2 oczu aby uznać że użytkownik patrzy.

        Args:
            frame: Obraz z kamery (numpy array BGR)

        Returns:
            bool: True jeśli wykryto oba oczy, False w przeciwnym razie
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        eyes_found = False
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = frame[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.05, 5)

            if len(eyes) >= 2:  # Oba oczy muszą być widoczne
                eyes_found = True
                for ex, ey, ew, eh in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )
        return eyes_found

    def update_status(self, eyes_detected):
        """FUNKCJA 2: Zatrzymuje reklamę gdy oglądający nie patrzy.

        Aktualizuje stan oglądania i zarządza pauzowaniem reklamy.
        Gdy użytkownik patrzy - reklama się odtwarza i licznik postępu rośnie.
        Gdy nie patrzy - reklama jest pauzowana i licznik się nie zwiększa.

        Args:
            eyes_detected (bool): Czy oczy zostały wykryte w bieżącej klatce
        """
        now = time.time()
        delta = now - self.last_time
        self.last_time = now

        if eyes_detected:
            self.is_watching = True
            self.not_watching_time = 0
            if self.ad_paused:
                self.ad_paused = False
            if not self.ad_paused:
                self.ad_watched_time += delta
        else:
            self.is_watching = False
            self.not_watching_time += delta
            if not self.ad_paused:
                self.ad_paused = True

    def draw_ui(self, frame):
        """Rysuje interfejs użytkownika na obrazie z kamery.

        FUNKCJA 3: Wyświetla alert gdy użytkownik nie patrzy przez >3 sekundy.

        Wyświetla:
        - Status oglądania (PATRZYSZ/NIE PATRZYSZ)
        - Status reklamy (PLAY/PAUZA)
        - Pasek postępu z procentami
        - Czerwony alert po 3 sekundach niepatrzenia
        - Komunikat o zakończeniu po obejrzeniu całości

        Args:
            frame: Obraz na którym rysowany jest interfejs (modyfikowany in-place)
        """
        h, w = frame.shape[:2]

        # Status
        status = "PATRZYSZ" if self.is_watching else "NIE PATRZYSZ"
        color = (0, 255, 0) if self.is_watching else (0, 0, 255)
        cv2.putText(
            frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        # Reklama
        ad_status = "PLAY" if not self.ad_paused else "PAUZA"
        cv2.putText(
            frame,
            f"Reklama: {ad_status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        # Pasek postępu
        progress = min(self.ad_watched_time / self.ad_duration, 1.0)
        bar_w = int(progress * (w - 40))
        cv2.rectangle(frame, (20, h - 50), (w - 20, h - 30), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 30), (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"{int(progress*100)}%",
            (20, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # FUNKCJA 3: Alert po określonym czasie (3 sekundy)
        if self.not_watching_time >= 3:
            cv2.rectangle(
                frame, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (0, 0, 255), -1
            )
            cv2.putText(
                frame,
                "UWAGA!",
                (w // 4 + 80, h // 2 - 20),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                frame,
                "Wroc do ogladania!",
                (w // 4 + 40, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        # Zakończenie
        if self.ad_watched_time >= self.ad_duration:
            cv2.rectangle(
                frame, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (0, 200, 0), -1
            )
            cv2.putText(
                frame,
                "KONIEC!",
                (w // 4 + 100, h // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (255, 255, 255),
                3,
            )

    def run(self):
        """Główna pętla programu - uruchamia platformę reklamową.

        Inicjalizuje kamerę, wyświetla instrukcje i rozpoczyna pętlę detekcji.
        W każdej iteracji:
        1. Pobiera klatkę z kamery
        2. Wykrywa oczy
        3. Aktualizuje stan reklamy
        4. Rysuje interfejs
        5. Obsługuje klawisze (Q - wyjście, R - reset)

        Obsługa klawiszy:
            Q - Zamyka program
            R - Resetuje reklamę do początku
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Błąd kamery!")
            return

        print("=" * 60)
        print("PLATFORMA REKLAMOWA")
        print("=" * 60)
        print("\nFUNKCJE:")
        print("1. Detekcja oczu (niebieski=twarz, zielony=oczy)")
        print("2. Zatrzymanie reklamy gdy nie patrzysz")
        print("3. Alert po 3 sekundach")
        print("\nQ - wyjście | R - reset\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            eyes = self.detect_eyes(frame)
            self.update_status(eyes)
            self.draw_ui(frame)

            cv2.imshow("Platforma Reklamowa", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.ad_watched_time = 0
                self.ad_paused = False
                self.not_watching_time = 0

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    AdPlatform().run()
