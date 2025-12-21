import cv2
import numpy as np
from collections import deque
import os

# Zamiast mediapipe, użyję OpenCV do detekcji gestów na podstawie koloru skóry


class SimpleGestureDetector:
    def __init__(self):
        self.gesture_history = deque(maxlen=5)
        self.last_action = ""
        self.action_cooldown = 0

    def detect_hand_contour(self, frame):
        """Detektuj ręce za pomocą koloru skóry"""
        # Konwersja na HSV (lepsze dla detekcji koloru skóry)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Zakres koloru skóry w HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Maska dla koloru skóry
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Zaburzenie szumów
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Znajdź kontury
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return mask, contours

    def count_fingers(self, contour):
        """Policz palce na podstawie konturu"""
        if cv2.contourArea(contour) < 1000:
            return 0

        # Aproksymuj kontur
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Liczba wierzchołków ~ liczba palców
        return len(approx)

    def detect_gesture_simple(self, contours, h, w):
        """Rozpoznaj gesty na podstawie liczby palców"""
        if not contours:
            return "UNKNOWN", None

        # Znajdź największy kontur (główną rękę)
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 1000:
            return "UNKNOWN", largest_contour

        # Policz palce
        fingers = self.count_fingers(largest_contour)

        # Mapowanie palców na gesty
        if fingers < 3:
            return "FIST", largest_contour  # Pięść
        elif fingers == 2:
            return "PEACE", largest_contour  # Dwa palce
        elif fingers == 3:
            return "OK_SIGN", largest_contour
        else:
            return "OPEN_HAND", largest_contour  # Otwarta dłoń

    def execute_gesture_action(self, gesture):
        """Wykonaj akcję na podstawie gestu"""
        if self.action_cooldown > 0:
            return None

        action = None

        if gesture == "OPEN_HAND":
            action = "Otwarta dłoń"
            self.action_cooldown = 15

        elif gesture == "FIST":
            action = "Pięść"
            self.action_cooldown = 15

        elif gesture == "PEACE":
            action = "Dwa palce (Peace)"
            self.action_cooldown = 15

        elif gesture == "OK_SIGN":
            action = "OK Sign"
            self.action_cooldown = 15

        if action:
            self.last_action = action

        return action

    def run(self):
        """Główna pętla programu"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Nie można otworzyć kamery!")
            return

        print("=" * 60)
        print("DETEKTOR GESTÓW - WERSJA UPROSZCZONA")
        print("=" * 60)
        print("\n📖 GESTY:\n")
        print("  ✊  FIST        - Pięść (< 3 palce)")
        print("  ✌️  PEACE       - Dwa palce (2 palce)")
        print("  👌 OK_SIGN     - OK (3 palce)")
        print("  ✋  OPEN_HAND   - Otwarta dłoń (>3 palce)")
        print("\n" + "=" * 60)
        print("Pokaż rękę do kamery! Naciśnij 'Q' aby wyjść\n")

        detector = SimpleGestureDetector()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Zmniejsz cooldown
            if detector.action_cooldown > 0:
                detector.action_cooldown -= 1

            # Detektuj rękę
            mask, contours = detector.detect_hand_contour(frame)

            # Rozpoznaj gest
            gesture, hand_contour = detector.detect_gesture_simple(contours, h, w)
            detector.gesture_history.append(gesture)

            # Wygładzona decyzja
            if detector.gesture_history:
                final_gesture = max(
                    set(detector.gesture_history), key=detector.gesture_history.count
                )
            else:
                final_gesture = gesture

            # Wykonaj akcję
            if final_gesture != "UNKNOWN":
                detector.execute_gesture_action(final_gesture)

            # Rysuj na ekranie
            if hand_contour is not None:
                cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)

                # Oblicz środek ręki
                M = cv2.moments(hand_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            # Wyświetl rozpoznany gest
            cv2.putText(
                frame,
                f"Gest: {final_gesture}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                2,
            )

            # Wyświetl ostatnią akcję
            if detector.last_action:
                cv2.putText(
                    frame,
                    f"✓ {detector.last_action}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
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

            cv2.imshow("Gesture Detection", frame)
            cv2.imshow("Mask (Skin Detection)", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n👋 Koniec!")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = SimpleGestureDetector()
    detector.run()
