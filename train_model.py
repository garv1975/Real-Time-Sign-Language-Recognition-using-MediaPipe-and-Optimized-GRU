import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# ── Load data ──────────────────────────────────────────────────────────────────
X = np.load("X.npy")
y = np.load("y.npy")

print(f"Loaded  X: {X.shape}")   # Expected: (N, 30, 126)
print(f"Loaded  y: {y.shape}")   # Expected: (N,)

# ── Encode labels ──────────────────────────────────────────────────────────────
le        = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat     = to_categorical(y_encoded)
num_classes = y_cat.shape[1]

np.save("labels.npy", le.classes_)
print(f"Classes ({num_classes}): {le.classes_}")

# ── Train / validation split ───────────────────────────────────────────────────
X_train, X_val, y_train, y_val, y_train_raw, y_val_raw = train_test_split(
    X, y_cat, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)
print(f"Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")

# ── Class weights ──────────────────────────────────────────────────────────────
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_raw),
    y=y_train_raw,
)
class_weight_dict = dict(enumerate(class_weights_arr))
print("Class weights computed.")

# ── GRU Model ──────────────────────────────────────────────────────────────────
# TWO-HAND SUPPORT:
# Input shape is now (SEQUENCE_LENGTH=30, features=126) instead of (30, 63).
# The extra 63 features represent the second (left) hand's wrist-normalised
# keypoints. When only one hand is present the left-hand slot is all zeros,
# so the model gracefully handles single-hand gestures too.
#
# Unit counts are unchanged from the one-hand GRU version; the wider input
# gives the first GRU layer more signal to work with, so no capacity boost
# is needed unless you add many new two-hand classes.

model = Sequential([
    # ── Block 1 ──────────────────────────────────────────────────────────────
    GRU(128, return_sequences=True, input_shape=(30, 126),  # ← 63 → 126
        recurrent_dropout=0.1),
    BatchNormalization(),
    Dropout(0.3),

    # ── Block 2 ──────────────────────────────────────────────────────────────
    GRU(256, return_sequences=True, recurrent_dropout=0.1),
    BatchNormalization(),
    Dropout(0.3),

    # ── Block 3 ──────────────────────────────────────────────────────────────
    GRU(128, return_sequences=False, recurrent_dropout=0.1),
    BatchNormalization(),
    Dropout(0.3),

    # ── Dense classifier ──────────────────────────────────────────────────────
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=40,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath="checkpoints/best_gru_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# ── Train ──────────────────────────────────────────────────────────────────────
print("\n🚀 Starting GRU training (two-hand, 126 features)...\n")
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)

# ── Save final model ───────────────────────────────────────────────────────────
model.save("model.h5")
print("\n✅ Training complete. model.h5 saved.")

# ── Evaluate ───────────────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n📊 Final Val Accuracy : {val_acc * 100:.2f}%")
print(f"   Final Val Loss     : {val_loss:.4f}")

best_epoch = np.argmax(history.history["val_accuracy"]) + 1
best_acc   = max(history.history["val_accuracy"])
print(f"\n🏆 Best Val Accuracy  : {best_acc * 100:.2f}%  at epoch {best_epoch}")
