import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
from datetime import datetime

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


class GestureTextEditor:
    def __init__(self):
        self.text = ""
        self.filename = "note.txt"
        self.gesture_history = deque(maxlen=5)
        self.last_action = ""
        self.action_cooldown = 0
        self.saved = True

    def is_open_hand(self, landmarks):
        """Wszystkie palce wyprostowane"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        all_extended = all(
            landmarks[tip].y < landmarks[pip].y
            for tip, pip in zip(finger_tips, finger_pips)
        )
        return all_extended

    def is_fist(self, landmarks):
        """Wszystkie palce zgięte - pięść"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        all_folded = all(
            landmarks[tip].y >= landmarks[pip].y
            for tip, pip in zip(finger_tips, finger_pips)
        )
        return all_folded

    def is_thumbs_up(self, landmarks):
        """Kciuk do góry"""
        thumb_up = landmarks[4].y < landmarks[2].y
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        others_folded = all(
            landmarks[tip].y >= landmarks[pip].y
            for tip, pip in zip(finger_tips, finger_pips)
        )
        return thumb_up and others_folded

    def is_peace(self, landmarks):
        """Dwa palce do góry"""
        index_extended = landmarks[8].y < landmarks[6].y
        middle_extended = landmarks[12].y < landmarks[10].y
        return index_extended and middle_extended

    def is_ok_sign(self, landmarks):
        """Kciuk + indeks w kółko"""
        thumb_index_distance = np.sqrt(
            (landmarks[4].x - landmarks[8].x) ** 2
            + (landmarks[4].y - landmarks[8].y) ** 2
        )
        return thumb_index_distance < 0.05

    def detect_gesture(self, landmarks):
        """Rozpoznanie gestu"""
        gestures = {
            "OPEN_HAND": self.is_open_hand(landmarks),
            "FIST": self.is_fist(landmarks),
            "THUMBS_UP": self.is_thumbs_up(landmarks),
            "PEACE": self.is_peace(landmarks),
            "OK_SIGN": self.is_ok_sign(landmarks),
        }

        for gesture, detected in gestures.items():
            if detected:
                return gesture

        return "UNKNOWN"

    def execute_gesture_action(self, gesture):
        """Wykonaj akcję na podstawie gestu"""
        if self.action_cooldown > 0:
            return None

        action = None

        if gesture == "THUMBS_UP":
            # Dodaj spację
            self.text += " "
            action = "Spacja dodana"
            self.saved = False
            self.action_cooldown = 15

        elif gesture == "PEACE":
            # Usuń ostatnią literę (backspace)
            if self.text:
                self.text = self.text[:-1]
            action = "Usunięto ostatni znak"
            self.saved = False
            self.action_cooldown = 15

        elif gesture == "OK_SIGN":
            # Dodaj nową linię
            self.text += "\n"
            action = "Nowa linia"
            self.saved = False
            self.action_cooldown = 15

        elif gesture == "FIST":
            # Wyczyść tekst
            self.text = ""
            action = "Tekst wyczyszczony"
            self.saved = False
            self.action_cooldown = 20

        elif gesture == "OPEN_HAND":
            # Zapisz plik
            self.save_file()
            action = "Plik zapisany!"
            self.saved = True
            self.action_cooldown = 30

        if action:
            self.last_action = action

        return action

    def save_file(self):
        """Zapisz tekst do pliku"""
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(self.text)

    def load_file(self):
        """Załaduj tekst z pliku"""
        if os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8") as f:
                self.text = f.read()
            self.saved = True

    def run(self):
        """Główna pętla programu"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Nie można otworzyć kamery!")
            return

        # Załaduj istniejący plik jeśli istnieje
        self.load_file()

        print("=" * 60)
        print("EDYTOR TEKSTU STEROWANY GESTAMI")
        print("=" * 60)
        print("\n📖 GESTY I ICH FUNKCJE:\n")
        print("  ✋  OPEN_HAND   -> ZAPISZ PLIK")
        print("  ✊  FIST        -> WYCZYŚĆ TEKST")
        print("  👍 THUMBS_UP   -> DODAJ SPACJĘ")
        print("  ✌️  PEACE       -> USUŃ ZNAK (BACKSPACE)")
        print("  👌 OK_SIGN     -> NOWA LINIA")
        print("\n" + "=" * 60)
        print(f"Plik: {self.filename}")
        print("Naciśnij 'Q' aby wyjść\n")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Zmniejsz cooldown
            if self.action_cooldown > 0:
                self.action_cooldown -= 1

            frame_count += 1

            # TŁO - zmień kolor w zależności od stanu
            if self.saved:
                bg_color = (0, 50, 0)  # Ciemnozielone - zapisano
            else:
                bg_color = (0, 0, 50)  # Ciemnoczerwone - niezapisane

            frame[:] = bg_color

            # Rysuj tekst w tle (zawartość edytora)
            text_lines = self.text.split("\n")
            y_offset = 100
            for i, line in enumerate(text_lines[-10:]):  # Pokaż ostatnie 10 linii
                if len(line) > 60:
                    line = line[:60] + "..."
                cv2.putText(
                    frame,
                    line,
                    (50, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 200, 100),
                    1,
                )

            # Detekcja dłoni i wyświetlenie
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Rysuj dłoń
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                    )

                    # Rozpoznaj gest
                    gesture = self.detect_gesture(hand_landmarks.landmark)
                    self.gesture_history.append(gesture)

                    # Wygładzona decyzja
                    if self.gesture_history:
                        final_gesture = max(
                            set(self.gesture_history), key=self.gesture_history.count
                        )
                    else:
                        final_gesture = gesture

                    # Wykonaj akcję
                    if final_gesture != "UNKNOWN":
                        self.execute_gesture_action(final_gesture)

                    # Wyświetl rozpoznany gest
                    cv2.putText(
                        frame,
                        f"Gest: {final_gesture}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        2,
                    )

            # Wyświetl ostatnią akcję
            if self.last_action:
                cv2.putText(
                    frame,
                    f"Akcja: {self.last_action}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # Wyświetl status
            status = "✓ ZAPISANO" if self.saved else "⚠ NIEZAPISANE"
            color = (0, 255, 0) if self.saved else (0, 0, 255)
            cv2.putText(
                frame, status, (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

            # Statystyki
            char_count = len(self.text)
            line_count = self.text.count("\n") + (1 if self.text else 0)
            cv2.putText(
                frame,
                f"Znaki: {char_count} | Linie: {line_count}",
                (50, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )

            # Instrukcja wyjścia
            cv2.putText(
                frame,
                "Nacisnij 'Q' aby wyjsc",
                (w - 300, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Gesture Text Editor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Zapisz przed wyjściem jeśli coś się zmieniło
        if not self.saved:
            self.save_file()
            print(f"\nTekst zapisany do: {self.filename}")

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nFinal text saved to: {self.filename}")
        print(f"Znaki: {len(self.text)}")
        print(f"Linie: {self.text.count(chr(10)) + (1 if self.text else 0)}")


if __name__ == "__main__":
    editor = GestureTextEditor()
    editor.run()
