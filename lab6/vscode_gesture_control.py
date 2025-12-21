import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import subprocess
import time
import sys

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


class VSCodeGestureController:
    def __init__(self):
        self.gesture_history = deque(maxlen=5)
        self.last_action = ""
        self.action_cooldown = 0
        self.vscode_focused = False

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

    def send_key(self, key_combo):
        """Wysłanie klawisza do VS Code (Linux)"""
        try:
            # xdotool - symulacja klawisza na Linuxie
            subprocess.run(["xdotool", "key", key_combo], check=False, timeout=1)
        except FileNotFoundError:
            print("⚠️  xdotool nie zainstalowany!")
            print("Zainstaluj: sudo apt-get install xdotool")
        except Exception as e:
            print(f"Błąd wysyłania klawisza: {e}")

    def send_text(self, text):
        """Wysłanie tekstu do VS Code"""
        try:
            subprocess.run(["xdotool", "type", text], check=False, timeout=1)
        except Exception as e:
            print(f"Błąd wysyłania tekstu: {e}")

    def execute_gesture_action(self, gesture):
        """Wykonaj akcję na podstawie gestu"""
        if self.action_cooldown > 0:
            return None

        action = None

        if gesture == "THUMBS_UP":
            # Dodaj spację
            self.send_text(" ")
            action = "Spacja dodana"
            self.action_cooldown = 15

        elif gesture == "PEACE":
            # Backspace - usuń ostatni znak
            self.send_key("BackSpace")
            action = "Usunięto znak"
            self.action_cooldown = 15

        elif gesture == "OK_SIGN":
            # Enter - nowa linia
            self.send_key("Return")
            action = "Nowa linia"
            self.action_cooldown = 15

        elif gesture == "FIST":
            # Ctrl+A - zaznacz wszystko
            self.send_key("ctrl+a")
            action = "Zaznaczono wszystko"
            self.action_cooldown = 20

        elif gesture == "OPEN_HAND":
            # Ctrl+S - zapisz plik
            self.send_key("ctrl+s")
            action = "Zapisano plik (Ctrl+S)"
            self.action_cooldown = 30

        if action:
            self.last_action = action

        return action

    def run(self):
        """Główna pętla programu"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Nie można otworzyć kamery!")
            return

        print("=" * 70)
        print("VS CODE GESTURE CONTROLLER")
        print("=" * 70)
        print("\n⚠️  WAŻNE: VS Code MUSI BYĆ AKTYWNYM OKNEM!")
        print("Klikni na edytor VS Code przed uruchomieniem gestów!\n")
        print("📖 GESTY I ICH FUNKCJE:\n")
        print("  ✋  OPEN_HAND   -> ZAPISZ PLIK (Ctrl+S)")
        print("  ✊  FIST        -> ZAZNACZ WSZYSTKO (Ctrl+A)")
        print("  👍 THUMBS_UP   -> DODAJ SPACJĘ")
        print("  ✌️  PEACE       -> USUŃ ZNAK (Backspace)")
        print("  👌 OK_SIGN     -> NOWA LINIA (Enter)")
        print("\n" + "=" * 70)
        print("Naciśnij 'Q' aby wyjść\n")

        input("Naciśnij ENTER gdy VS Code będzie aktywny... ")
        print("🎬 Start! Życzę powodzenia!\n")

        time.sleep(1)  # Daj chwilę na przełączenie okna

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

            # TŁO - ciemne
            frame[:] = (20, 20, 30)

            # Nagłówek
            cv2.putText(
                frame,
                "VS CODE GESTURE CONTROL",
                (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
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
                        (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "Czekam na rękę...",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (100, 100, 100),
                    2,
                )

            # Wyświetl ostatnią akcję
            if self.last_action:
                cv2.putText(
                    frame,
                    f"✓ {self.last_action}",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            # Instrukcje
            cv2.putText(
                frame,
                "Q - wyjscie",
                (w - 250, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                1,
            )

            # Cooldown indicator
            if self.action_cooldown > 0:
                cooldown_pct = int((self.action_cooldown / 30) * 100)
                cv2.putText(
                    frame,
                    f"Cooldown: {cooldown_pct}%",
                    (50, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 100, 255),
                    2,
                )

            cv2.imshow("VS Code Gesture Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n👋 Koniec!")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = VSCodeGestureController()
    controller.run()
