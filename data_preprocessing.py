import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from collections import Counter

mp_hands = mp.solutions.hands

# ── Dataset config ─────────────────────────────────────────────────────────────
DATA_PATH       = "ASL_dynamic"
SEQUENCE_LENGTH = 30   # frames per sequence (matches training videos ~30 fps)

# Dataset: 31 classes × 10 videos → small corpus, so augment more aggressively.
# Classes with very few UNIQUE source videos get extra copies.
UNDERREPRESENTED = {"YES", "NO", "L", "C", "K"}

# Augmentation multipliers
AUG_FACTOR_NORMAL       = 5   # 1 original + 4 augmented
AUG_FACTOR_UNDERREPR    = 8   # 1 original + 7 augmented

# ── Keypoint extraction ────────────────────────────────────────────────────────
# TWO-HAND SUPPORT:
# We always extract keypoints for BOTH hands. If a hand is absent, we return
# zeros for that hand's 63 values. Final vector = 126 (left_63 + right_63).
#
# Hand assignment strategy: MediaPipe labels hands as "Left" / "Right" from the
# *camera's* perspective (i.e. mirrored). We sort by that label so the feature
# vector is always [LEFT_hand_63 | RIGHT_hand_63] regardless of detection order.
# For single-hand gestures, only one half will be non-zero — the model learns
# to ignore the zero half automatically.

def extract_keypoints(results):
    """
    Extract wrist-normalised keypoints for both hands from a MediaPipe result.
    Returns a float32 array of shape (126,) — [left_63 | right_63].
    """
    left_kps  = np.zeros(63, dtype=np.float32)
    right_kps = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" or "Right"
            base  = hand_landmarks.landmark[0]
            kps   = []
            for lm in hand_landmarks.landmark:
                kps.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])
            kps = np.array(kps, dtype=np.float32)

            if label == "Left":
                left_kps = kps
            else:
                right_kps = kps

    return np.concatenate([left_kps, right_kps])   # shape (126,)


# ── Sequence utilities ─────────────────────────────────────────────────────────
def pad_sequence(sequence):
    """Repeat the last frame until the sequence reaches SEQUENCE_LENGTH."""
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(sequence[-1].copy())
    return sequence[:SEQUENCE_LENGTH]


def augment_noise(sequence, noise_std=0.005):
    """Add per-frame Gaussian noise to keypoints."""
    seq   = np.array(sequence, dtype=np.float32)
    noise = np.random.normal(0, noise_std, seq.shape).astype(np.float32)
    return seq + noise


def temporal_warp(sequence):
    """
    Randomly shift frame indices to simulate speed variation
    (sub-sample / forward-interpolate by up to ±3 frames).
    """
    seq     = np.array(sequence, dtype=np.float32)
    n       = len(seq)
    shift   = np.random.randint(-3, 4)
    indices = np.clip(np.arange(n) + shift, 0, n - 1)
    return seq[indices]


def random_scale(sequence, scale_range=(0.9, 1.1)):
    """
    Uniformly scale all keypoint coordinates.
    Simulates slight distance variation from the camera.
    """
    scale = np.random.uniform(*scale_range)
    return np.array(sequence, dtype=np.float32) * scale


def random_flip_hands(sequence):
    """
    Mirror the x-axis AND swap left/right hand slots.
    A horizontal flip of the scene turns a right-hand gesture into a left-hand
    one, so we must also exchange the two 63-value halves.
    Shape: (T, 126)  →  [left_63 | right_63]  →  [-right_x | -left_x]
    """
    seq = np.array(sequence, dtype=np.float32)          # (T, 126)
    left_half  = seq[:, :63].copy()
    right_half = seq[:, 63:].copy()

    # Negate x-coordinates (every 3rd value starting at 0) in each half
    left_half[:, 0::3]  = -left_half[:, 0::3]
    right_half[:, 0::3] = -right_half[:, 0::3]

    # Swap: what was right becomes left and vice-versa
    flipped = np.concatenate([right_half, left_half], axis=1)
    return flipped


# ── Main processing loop ───────────────────────────────────────────────────────
def process_dataset():
    X, y   = [], []
    labels = sorted(os.listdir(DATA_PATH))

    print(f"\nFound {len(labels)} classes: {labels}\n")

    # static_image_mode=True → each frame treated independently (no temporal carry-over).
    # max_num_hands=2 → detect both hands.
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,            # ← changed from 1 to 2
        min_detection_confidence=0.5
    ) as hands:

        for label in tqdm(labels, desc="Processing classes"):
            label_path  = os.path.join(DATA_PATH, label)
            video_files = [f for f in os.listdir(label_path) if f.endswith(".avi")]

            if not video_files:
                print(f"  WARNING: No .avi files found for '{label}'")
                continue

            class_sequences = []

            for video_file in tqdm(video_files, desc=f"  {label}", leave=False):
                video_path = os.path.join(label_path, video_file)
                cap        = cv2.VideoCapture(video_path)

                sequence = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results   = hands.process(image_rgb)
                    kp        = extract_keypoints(results)   # (126,)
                    sequence.append(kp)
                    if len(sequence) >= SEQUENCE_LENGTH:
                        break
                cap.release()

                if len(sequence) == 0:
                    print(f"    SKIP (empty): {video_file}")
                    continue

                sequence = pad_sequence(sequence)
                class_sequences.append(np.array(sequence, dtype=np.float32))

            if not class_sequences:
                print(f"  WARNING: No valid sequences for '{label}'")
                continue

            aug_factor = (
                AUG_FACTOR_UNDERREPR if label.upper() in UNDERREPRESENTED
                else AUG_FACTOR_NORMAL
            )

            for seq in class_sequences:
                # 1. Original sequence
                X.append(seq)
                y.append(label)

                # 2. Augmented copies
                for _ in range(aug_factor - 1):
                    aug = seq.copy()

                    # Noise (always applied)
                    noise_std = np.random.uniform(0.003, 0.010)
                    aug       = augment_noise(aug, noise_std=noise_std)

                    # Temporal warp (60 % chance)
                    if np.random.random() < 0.6:
                        aug = temporal_warp(aug)

                    # Scale jitter (50 % chance)
                    if np.random.random() < 0.5:
                        aug = random_scale(aug)

                    # Hand flip (30 % chance) — now swaps left/right slots too
                    if np.random.random() < 0.3:
                        aug = random_flip_hands(aug)

                    X.append(aug)
                    y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # ── Summary ────────────────────────────────────────────────────────────────
    dist = Counter(y)
    print("\n--- Final Class Distribution ---")
    for lbl in sorted(dist.keys()):
        print(f"  {lbl:12s}: {dist[lbl]:4d} samples")
    print(f"  {'TOTAL':12s}: {len(y):4d} samples")
    print("--------------------------------\n")

    return X, y


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    X, y = process_dataset()
    np.save("X.npy", X)
    np.save("y.npy", y)
    print("Data saved to X.npy and y.npy")
    print("X shape:", X.shape)   # Expected: (N, 30, 126)
    print("y shape:", y.shape)   # Expected: (N,)
