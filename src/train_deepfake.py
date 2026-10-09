import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =========================
# Configuration
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

TRAIN_DIR = "data/deepfake/train"
VAL_DIR = "data/deepfake/val"
TEST_DIR = "data/deepfake/test"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# Load datasets
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Save class names BEFORE prefetch
class_names = train_ds.class_names

print("\nClass names:", class_names)

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# Data augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
])

# =========================
# EfficientNet-B0
# =========================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# =========================
# Callbacks
# =========================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=1,
        min_lr=1e-7
    ),

    ModelCheckpoint(
        "models/best_deepfake_efficientnet.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =========================
# Stage 1: Train classifier
# =========================

print("\n========== STAGE 1: TRAINING ==========\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# =========================
# Stage 2: Fine-tuning
# =========================

print("\n========== STAGE 2: FINE-TUNING ==========\n")

base_model.trainable = True

# Freeze earlier layers, fine-tune later layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# =========================
# Final test evaluation
# =========================

print("\n========== TEST EVALUATION ==========\n")

test_results = model.evaluate(test_ds, verbose=1)

for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

# =========================
# Predictions
# =========================

y_true = np.concatenate([
    y.numpy().flatten()
    for _, y in test_ds
])

y_prob = model.predict(test_ds).flatten()

y_pred = (y_prob >= 0.5).astype(int)

# =========================
# Classification report
# =========================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# =========================
# Confusion matrix
# =========================

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(y_true, y_pred)

print(cm)

# =========================
# ROC-AUC
# =========================

auc = roc_auc_score(y_true, y_prob)

print(f"\nROC-AUC: {auc:.4f}")

# =========================
# Save final model
# =========================

model.save("models/deepfake_efficientnet_final.keras")

print("\nModel saved successfully!")
print("models/deepfake_efficientnet_final.keras")