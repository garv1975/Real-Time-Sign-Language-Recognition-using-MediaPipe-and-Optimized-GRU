import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from tensorflow.keras.models import load_model

# ── Load model ─────────────────────────────────────────────
model  = load_model("model.h5")
labels = np.load("labels.npy")

mp_hands = mp.solutions.hands

# ── Parameters ─────────────────────────────────────────────
SEQUENCE_LENGTH   = 30
CONFIDENCE_THRESH = 0.65
SMOOTHING_WINDOW  = 12
COOLDOWN_FRAMES   = 20
CLEAR_THRESH_SEC  = 4

# ── Buffers ────────────────────────────────────────────────
sequence     = deque(maxlen=SEQUENCE_LENGTH)
predictions  = deque(maxlen=SMOOTHING_WINDOW)

# ── Sentence State ─────────────────────────────────────────
sentence        = []
last_word       = ""
cooldown        = 0
last_hand_time  = time.time()

# ── Keypoint extraction ────────────────────────────────────
# TWO-HAND SUPPORT:
# We build a 126-dim vector = [left_63 | right_63].
# MediaPipe's "Left"/"Right" labels are from the camera's perspective
# (mirrored view). Zero-padding is used when a hand is absent, matching
# exactly what was done during preprocessing.
def extract_keypoints(results):
    """
    Returns a float32 array of shape (126,) — [left_63 | right_63].
    Missing hands are represented as zeros.
    """
    left_kps  = np.zeros(63, dtype=np.float32)
    right_kps = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label   # "Left" or "Right"
            base  = hand_landmarks.landmark[0]
            kps   = []
            for lm in hand_landmarks.landmark:
                kps.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])
            kps = np.array(kps, dtype=np.float32)

            if label == "Left":
                left_kps = kps
            else:
                right_kps = kps

    return np.concatenate([left_kps, right_kps])   # (126,)


cap = cv2.VideoCapture(0)

print("✅ Continuous sentence detection started (two-hand mode)")
print("q = quit | c = clear | space = allow repeat\n")

with mp_hands.Hands(
    max_num_hands=2,                # ← detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        hand_detected = results.multi_hand_landmarks is not None

        if hand_detected:
            last_hand_time = time.time()

            # Draw landmarks for ALL detected hands (one or two)
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # "Left" / "Right"
                # Color-code: green for right hand, blue for left hand
                color = (0, 255, 0) if hand_label == "Right" else (255, 150, 0)

                for lm in hand_landmarks.landmark:
                    x = int((1 - lm.x) * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(display, (x, y), 3, color, -1)

        # ── Build keypoint vector (126,) ───────────────────
        keypoints = extract_keypoints(results)

        # ── Sequence ───────────────────────────────────────
        sequence.append(keypoints)

        # ── Prediction ─────────────────────────────────────
        current_word = ""
        confidence   = 0

        if len(sequence) == SEQUENCE_LENGTH:
            inp = np.expand_dims(np.array(sequence), axis=0)   # (1, 30, 126)
            res = model.predict(inp, verbose=0)[0]

            idx  = np.argmax(res)
            conf = float(res[idx])

            if conf > CONFIDENCE_THRESH:
                predictions.append(labels[idx])

                if len(predictions) == SMOOTHING_WINDOW:
                    voted        = max(set(predictions), key=predictions.count)
                    current_word = voted
                    confidence   = conf * 100

                    # ── Add to sentence ───────────────────
                    if cooldown == 0 and current_word != last_word:
                        sentence.append(current_word)
                        last_word = current_word
                        cooldown  = COOLDOWN_FRAMES
                        print("Sentence:", " ".join(sentence))
            else:
                predictions.clear()

        # ── Cooldown ───────────────────────────────────────
        if cooldown > 0:
            cooldown -= 1

        # ── Auto clear ─────────────────────────────────────
        if not hand_detected and (time.time() - last_hand_time > CLEAR_THRESH_SEC):
            sentence.clear()
            last_word = ""

        # ── Hand count indicator ───────────────────────────
        num_hands = len(results.multi_hand_landmarks) if hand_detected else 0
        hand_info = f"Hands: {num_hands}/2"

        # ── UI TOP (Current word) ──────────────────────────
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        if current_word:
            cv2.putText(display, current_word, (20, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 120), 3)

        # Hand count badge (top-right)
        cv2.putText(display, hand_info,
                    (display.shape[1] - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 255) if num_hands == 2 else (180, 180, 180), 2)

        # ── UI BOTTOM (Sentence) ───────────────────────────
        cv2.rectangle(display,
                      (0, display.shape[0] - 100),
                      (display.shape[1], display.shape[0]),
                      (0, 0, 0), -1)

        text = " ".join(sentence[-6:])
        cv2.putText(display, text,
                    (20, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)

        # ── Progress Bar ───────────────────────────────────
        fill = int(display.shape[1] * (len(sequence) / SEQUENCE_LENGTH))
        cv2.rectangle(display, (0, 78), (fill, 82), (0, 200, 100), -1)

        cv2.imshow("ASL Sentence Detection", display)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence.clear()
            last_word = ""
        elif key == ord(' '):
            last_word = ""   # allow same word again

cap.release()
cv2.destroyAllWindows()
