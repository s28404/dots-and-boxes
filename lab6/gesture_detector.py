import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


# Gesty: nazwa -> funkcja do sprawdzenia warunku
def is_open_hand(landmarks):
    """Wszystkie palce wyprostowane"""
    # Sprawdzamy czy wszystkie palce są 'wzniesione' (wysoko)
    # Warunek: jeśli wszystkie końce palców są wyżej niż ich podstawy
    finger_tips = [4, 8, 12, 16, 20]  # Indeksy końców palców
    finger_pips = [3, 6, 10, 14, 18]  # Indeksy PIP (drugi człon)

    all_extended = all(
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    )
    return all_extended


def is_fist(landmarks):
    """Wszystkie palce zgięte - pięść"""
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]

    all_folded = all(
        landmarks[tip].y >= landmarks[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    )
    return all_folded


def is_thumbs_up(landmarks):
    """Kciuk do góry, reszta zgięta"""
    # Kciuk (4) powyżej handshake point (9)
    # Pozostałe palce zgięte
    thumb_up = landmarks[4].y < landmarks[2].y  # Kciuk wysoko

    # Pozostałe palce zgięte
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    others_folded = all(
        landmarks[tip].y >= landmarks[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    )

    return thumb_up and others_folded


def is_peace(landmarks):
    """Dwa palce do góry (indeks + środkowy), reszta zgięta"""
    # Indeks (8) i środkowy (12) wyprostowane
    # Kciuk, serdeczny, mały zgięte
    index_extended = landmarks[8].y < landmarks[6].y
    middle_extended = landmarks[12].y < landmarks[10].y

    # Reszta zgięta
    thumb_folded = landmarks[4].y >= landmarks[3].y
    ring_folded = landmarks[16].y >= landmarks[14].y
    pinky_folded = landmarks[20].y >= landmarks[18].y

    return (
        index_extended
        and middle_extended
        and thumb_folded
        and ring_folded
        and pinky_folded
    )


def is_ok_sign(landmarks):
    """Kciuk + indeks w kółko, reszta wyprostowana"""
    # Kciuk i indeks blisko siebie (tworz ą kółko)
    thumb_index_distance = np.sqrt(
        (landmarks[4].x - landmarks[8].x) ** 2 + (landmarks[4].y - landmarks[8].y) ** 2
    )

    # Pozostałe palce wyprostowane
    middle_extended = landmarks[12].y < landmarks[10].y
    ring_extended = landmarks[16].y < landmarks[14].y
    pinky_extended = landmarks[20].y < landmarks[18].y

    return (
        thumb_index_distance < 0.05
        and middle_extended
        and ring_extended
        and pinky_extended
    )


def detect_gesture(landmarks):
    """Główna funkcja do rozpoznania gestu"""
    gestures = {
        "OPEN_HAND": is_open_hand(landmarks),
        "FIST": is_fist(landmarks),
        "THUMBS_UP": is_thumbs_up(landmarks),
        "PEACE": is_peace(landmarks),
        "OK_SIGN": is_ok_sign(landmarks),
    }

    # Zwracamy gest z największym pewnością (True najpierw)
    for gesture, detected in gestures.items():
        if detected:
            return gesture

    return "UNKNOWN"


def main():
    # Otwarcie kamery
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Nie można otworzyć kamery!")
        return

    # Historia gestów (do wygładzania)
    gesture_history = deque(maxlen=5)

    print("Program do detekcji gestów")
    print("Naciśnij 'q' aby wyjść")
    print("\nGotowe gesty:")
    print("  ✋  OPEN_HAND   - ręka otwarta")
    print("  ✊  FIST        - pięść")
    print("  👍 THUMBS_UP   - kciuk do góry")
    print("  ✌️  PEACE       - dwa palce")
    print("  👌 OK_SIGN     - kółeczko z kciuka i indeksu")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Przerzucenie obrazu (lustro)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Konwersja BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detekcja dłoni
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Rysowanie landmarków
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

                # Rozpoznanie gestu
                gesture = detect_gesture(hand_landmarks.landmark)
                gesture_history.append(gesture)

                # Wygładzona decyzja (most common gesture in history)
                if gesture_history:
                    final_gesture = max(set(gesture_history), key=gesture_history.count)
                else:
                    final_gesture = gesture

                # Wyświetlenie wyniku
                cv2.putText(
                    frame,
                    f"Gest: {final_gesture}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )
        else:
            cv2.putText(
                frame,
                "Brak dłoni w kadrze",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Wyświetlenie FPS
        cv2.putText(
            frame,
            f"Nacisnij 'q' aby wyjsc",
            (50, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Gesture Detection", frame)

        # Wyjście na 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
