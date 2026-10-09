import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

MARGIN = 1.0
SEED = 42

TRAIN_CSV = "data/verification/pairs/train_pairs.csv"
VAL_CSV = "data/verification/pairs/val_pairs.csv"
TEST_CSV = "data/verification/pairs/test_pairs.csv"

MODEL_PATH = "models/siamese_efficientnet.keras"
RESULT_PATH = "results/siamese_cross_age_results.csv"
PREDICTION_PATH = "results/siamese_test_predictions.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# CUSTOM L2 NORMALIZATION LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable()
class L2Normalization(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(
            inputs,
            axis=1
        )

    def get_config(self):
        return super().get_config()


# ============================================================
# CUSTOM EUCLIDEAN DISTANCE LAYER
# ============================================================

@tf.keras.utils.register_keras_serializable()
class EuclideanDistance(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):

        embedding1, embedding2 = inputs

        difference = embedding1 - embedding2

        squared_difference = tf.square(
            difference
        )

        summed = tf.reduce_sum(
            squared_difference,
            axis=1,
            keepdims=True
        )

        distance = tf.sqrt(
            summed + 1e-10
        )

        return distance

    def get_config(self):
        return super().get_config()


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

@tf.keras.utils.register_keras_serializable()
def contrastive_loss(y_true, y_pred):

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_true = tf.reshape(
        y_true,
        (-1, 1)
    )

    positive_loss = (
        y_true *
        tf.square(y_pred)
    )

    negative_loss = (
        (1.0 - y_true) *
        tf.square(
            tf.maximum(
                MARGIN - y_pred,
                0.0
            )
        )
    )

    loss = (
        positive_loss +
        negative_loss
    )

    return tf.reduce_mean(loss)


# ============================================================
# LOAD CSV FILES
# ============================================================

print("\n========================================")
print("LOADING VERIFICATION PAIRS")
print("========================================\n")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Training pairs   :", len(train_df))
print("Validation pairs :", len(val_df))
print("Test pairs       :", len(test_df))


# ============================================================
# CHECK CSV COLUMNS
# ============================================================

required_columns = [
    "image1",
    "image2",
    "label"
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing column '{column}' in training CSV."
        )

    if column not in val_df.columns:
        raise ValueError(
            f"Missing column '{column}' in validation CSV."
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing column '{column}' in test CSV."
        )


# ============================================================
# CONVERT RELATIVE PATHS TO NORMAL PATHS
# ============================================================

def normalize_path(path):

    path = str(path)

    path = path.replace("\\", os.sep)
    path = path.replace("/", os.sep)

    return path


train_df["image1"] = train_df["image1"].apply(
    normalize_path
)

train_df["image2"] = train_df["image2"].apply(
    normalize_path
)

val_df["image1"] = val_df["image1"].apply(
    normalize_path
)

val_df["image2"] = val_df["image2"].apply(
    normalize_path
)

test_df["image1"] = test_df["image1"].apply(
    normalize_path
)

test_df["image2"] = test_df["image2"].apply(
    normalize_path
)


# ============================================================
# VERIFY FILES
# ============================================================

def check_files(df, name):

    missing = []

    for path in df["image1"]:
        if not os.path.exists(path):
            missing.append(path)

    for path in df["image2"]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:

        print(
            f"\nERROR: {len(missing)} image files "
            f"were not found in {name}."
        )

        print("Example missing file:")
        print(missing[0])

        raise FileNotFoundError(
            "Some image files referenced by the CSV do not exist."
        )

    print(
        f"{name}: all image files found."
    )


check_files(train_df, "Training")
check_files(val_df, "Validation")
check_files(test_df, "Test")


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image_bytes = tf.io.read_file(path)

    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(
    dataframe,
    training=False
):

    image1_paths = dataframe[
        "image1"
    ].values

    image2_paths = dataframe[
        "image2"
    ].values

    labels = dataframe[
        "label"
    ].values.astype(
        np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image1_paths,
            image2_paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    def process(
        image1_path,
        image2_path,
        label
    ):

        image1 = load_image(
            image1_path
        )

        image2 = load_image(
            image2_path
        )

        return (
            {
                "image1": image1,
                "image2": image2
            },
            label
        )

    dataset = dataset.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


print("\nCreating TensorFlow datasets...")

train_ds = create_dataset(
    train_df,
    training=True
)

val_ds = create_dataset(
    val_df,
    training=False
)

test_ds = create_dataset(
    test_df,
    training=False
)

print("Datasets created successfully.")


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.03
        ),

        layers.RandomZoom(
            0.08
        )
    ],
    name="image_augmentation"
)


# ============================================================
# BUILD EFFICIENTNET-B0
# ============================================================

print("\n========================================")
print("BUILDING EFFICIENTNET-B0")
print("========================================\n")

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(
        224,
        224,
        3
    )
)

# Initially freeze EfficientNet
base_model.trainable = False


# ============================================================
# EMBEDDING NETWORK
# ============================================================

embedding_input = layers.Input(
    shape=(
        224,
        224,
        3
    ),
    name="embedding_input"
)

x = augmentation(
    embedding_input
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(
    x
)

x = layers.Dense(
    256,
    activation="relu",
    name="embedding_dense"
)(
    x
)

x = layers.Dropout(
    0.30
)(
    x
)

embedding_output = L2Normalization(
    name="embedding_normalization"
)(
    x
)

embedding_model = Model(
    inputs=embedding_input,
    outputs=embedding_output,
    name="efficientnet_embedding"
)


# ============================================================
# SIAMESE NETWORK
# ============================================================

print("\nBuilding Siamese network...")

input1 = layers.Input(
    shape=(
        224,
        224,
        3
    ),
    name="image1"
)

input2 = layers.Input(
    shape=(
        224,
        224,
        3
    ),
    name="image2"
)

embedding1 = embedding_model(
    input1
)

embedding2 = embedding_model(
    input2
)

distance = EuclideanDistance(
    name="euclidean_distance"
)(
    [
        embedding1,
        embedding2
    ]
)

siamese_model = Model(
    inputs=[
        input1,
        input2
    ],
    outputs=distance,
    name="siamese_efficientnet"
)


# ============================================================
# STAGE 1 COMPILATION
# ============================================================

siamese_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=contrastive_loss
)


print("\n========================================")
print("SIAMESE MODEL SUMMARY")
print("========================================\n")

siamese_model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=1,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n========================================")
print("STAGE 1: FROZEN EFFICIENTNET TRAINING")
print("========================================\n")

history_stage1 = siamese_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# STAGE 2 FINE-TUNING
# ============================================================

print("\n========================================")
print("STAGE 2: EFFICIENTNET FINE-TUNING")
print("========================================\n")

base_model.trainable = True


# Freeze earlier layers
for layer in base_model.layers[:-30]:

    layer.trainable = False


# Unfreeze final 30 layers
for layer in base_model.layers[-30:]:

    layer.trainable = True


siamese_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss=contrastive_loss
)


history_stage2 = siamese_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\n========================================")
print("LOADING BEST MODEL")
print("========================================\n")

best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "L2Normalization":
            L2Normalization,

        "EuclideanDistance":
            EuclideanDistance,

        "contrastive_loss":
            contrastive_loss
    },
    safe_mode=False
)

print("Best model loaded successfully.")


# ============================================================
# GET DISTANCES AND LABELS
# ============================================================

def get_distances(
    model,
    dataset
):

    all_distances = []
    all_labels = []

    for batch_inputs, batch_labels in dataset:

        batch_distances = model.predict(
            batch_inputs,
            verbose=0
        )

        batch_distances = np.asarray(
            batch_distances
        ).reshape(-1)

        batch_labels = np.asarray(
            batch_labels
        ).reshape(-1)

        all_distances.extend(
            batch_distances
        )

        all_labels.extend(
            batch_labels
        )

    return (
        np.asarray(
            all_distances,
            dtype=np.float32
        ),

        np.asarray(
            all_labels,
            dtype=np.int32
        )
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n========================================")
print("VALIDATION PREDICTIONS")
print("========================================\n")

val_distances, val_labels = get_distances(
    best_model,
    val_ds
)

print(
    "Validation samples:",
    len(val_labels)
)


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

print("\n========================================")
print("FINDING BEST VERIFICATION THRESHOLD")
print("========================================\n")

best_threshold = 0.50
best_validation_accuracy = 0.0

thresholds = np.arange(
    0.05,
    1.001,
    0.01
)

for threshold in thresholds:

    validation_predictions = (
        val_distances <= threshold
    ).astype(int)

    validation_accuracy = accuracy_score(
        val_labels,
        validation_predictions
    )

    if (
        validation_accuracy
        >
        best_validation_accuracy
    ):

        best_validation_accuracy = (
            validation_accuracy
        )

        best_threshold = (
            threshold
        )


print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.4f}"
)

print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n========================================")
print("TEST EVALUATION")
print("========================================\n")

test_distances, test_labels = get_distances(
    best_model,
    test_ds
)

test_predictions = (
    test_distances <= best_threshold
).astype(int)


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

similarity_scores = (
    -test_distances
)

roc_auc = roc_auc_score(
    test_labels,
    similarity_scores
)

cm = confusion_matrix(
    test_labels,
    test_predictions
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n========================================")
print("CROSS-AGE VERIFICATION RESULTS")
print("========================================\n")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print(
    f"\nVerification threshold: "
    f"{best_threshold:.2f}"
)


# ============================================================
# SAVE METRICS
# ============================================================

results_df = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "verification_threshold",
            "validation_accuracy"
        ],

        "value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            best_threshold,
            best_validation_accuracy
        ]
    }
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test_df.copy()

prediction_df[
    "distance"
] = test_distances

prediction_df[
    "prediction"
] = test_predictions

prediction_df[
    "correct"
] = (
    prediction_df["label"]
    ==
    prediction_df["prediction"]
)

prediction_df.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("SIAMESE TRAINING AND EVALUATION COMPLETE")
print("========================================\n")

print(
    "Model saved:"
)

print(
    MODEL_PATH
)

print(
    "\nMetrics saved:"
)

print(
    RESULT_PATH
)

print(
    "\nTest predictions saved:"
)

print(
    PREDICTION_PATH
)

print("\nProject Stage 2 completed successfully.")