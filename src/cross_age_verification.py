import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.applications import EfficientNetB0


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = (224, 224)

METADATA_PATH = "data/verification/agedb/train/metadata.csv"
IMAGE_ROOT = "data/verification/agedb/train"

MODEL_PATH = "models/deepfake_efficientnet_final.keras"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42

# Number of positive and negative pairs
NUM_PAIRS = 1000


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Load metadata
# ============================================================

print("\n========== LOADING AGEDB METADATA ==========\n")

df = pd.read_csv(METADATA_PATH)

print("Total images:", len(df))
print("Total identities:", df["identity"].nunique())

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# Prepare valid images
# ============================================================

df["full_path"] = df["file_name"].apply(
    lambda x: os.path.join(IMAGE_ROOT, x.replace("/", os.sep))
)

# Keep only images that actually exist
df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

print("\nImages found on disk:", len(df))
print("Identities found:", df["identity"].nunique())


# ============================================================
# Create cross-age positive pairs
# ============================================================

print("\n========== CREATING PAIRS ==========\n")

positive_pairs = []

grouped = df.groupby("identity")

for identity, group in grouped:

    # Need at least two images of the same person
    if len(group) < 2:
        continue

    rows = list(group.iterrows())

    # Try several combinations
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):

            idx1, row1 = rows[i]
            idx2, row2 = rows[j]

            # Only use genuinely different ages
            if row1["age"] == row2["age"]:
                continue

            positive_pairs.append({
                "image1": row1["full_path"],
                "image2": row2["full_path"],
                "identity1": row1["identity"],
                "identity2": row2["identity"],
                "age1": row1["age"],
                "age2": row2["age"],
                "label": 1
            })

# Shuffle
random.shuffle(positive_pairs)

positive_pairs = positive_pairs[:NUM_PAIRS]

print("Positive pairs:", len(positive_pairs))


# ============================================================
# Create negative pairs
# ============================================================

negative_pairs = []

identities = df["identity"].unique().tolist()

while len(negative_pairs) < NUM_PAIRS:

    id1, id2 = random.sample(identities, 2)

    group1 = df[df["identity"] == id1]
    group2 = df[df["identity"] == id2]

    row1 = group1.sample(1, random_state=random.randint(0, 100000)).iloc[0]
    row2 = group2.sample(1, random_state=random.randint(0, 100000)).iloc[0]

    negative_pairs.append({
        "image1": row1["full_path"],
        "image2": row2["full_path"],
        "identity1": row1["identity"],
        "identity2": row2["identity"],
        "age1": row1["age"],
        "age2": row2["age"],
        "label": 0
    })

print("Negative pairs:", len(negative_pairs))


# ============================================================
# Combine pairs
# ============================================================

pairs = positive_pairs + negative_pairs

random.shuffle(pairs)

pairs_df = pd.DataFrame(pairs)

pairs_path = os.path.join(
    RESULTS_DIR,
    "cross_age_pairs.csv"
)

pairs_df.to_csv(pairs_path, index=False)

print("\nPairs saved to:")
print(pairs_path)

print("\nTotal pairs:", len(pairs_df))


# ============================================================
# Load trained EfficientNet model
# ============================================================

print("\n========== LOADING EFFICIENTNET MODEL ==========\n")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Create embedding model
# ============================================================

# Use the layer before the final sigmoid classifier
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-3].output
)

print("Embedding model created.")

print("Embedding shape:", embedding_model.output_shape)


# ============================================================
# Image preprocessing
# ============================================================

def load_image(path):

    image = tf.keras.utils.load_img(
        path,
        target_size=IMG_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# Generate embeddings
# ============================================================

print("\n========== GENERATING EMBEDDINGS ==========\n")

embedding_cache = {}


def get_embedding(path):

    if path in embedding_cache:
        return embedding_cache[path]

    image = load_image(path)

    embedding = embedding_model.predict(
        image,
        verbose=0
    )[0]

    embedding_cache[path] = embedding

    return embedding


# ============================================================
# Calculate cosine similarity
# ============================================================

print("Calculating cosine similarities...")

similarities = []
labels = []

for index, row in pairs_df.iterrows():

    emb1 = get_embedding(row["image1"])
    emb2 = get_embedding(row["image2"])

    similarity = cosine_similarity(
        emb1.reshape(1, -1),
        emb2.reshape(1, -1)
    )[0][0]

    similarities.append(similarity)
    labels.append(row["label"])

    if (index + 1) % 100 == 0:
        print(
            f"Processed {index + 1}/{len(pairs_df)} pairs"
        )


similarities = np.array(similarities)
labels = np.array(labels)


# ============================================================
# Find best threshold
# ============================================================

print("\n========== FINDING BEST THRESHOLD ==========\n")

best_threshold = 0.5
best_accuracy = 0

thresholds = np.arange(
    0.10,
    1.00,
    0.01
)

for threshold in thresholds:

    predictions = (
        similarities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold


print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Best accuracy: {best_accuracy:.4f}"
)


# ============================================================
# Final verification predictions
# ============================================================

predictions = (
    similarities >= best_threshold
).astype(int)


# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(
    labels,
    predictions
)

precision = precision_score(
    labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    labels,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    labels,
    similarities
)

cm = confusion_matrix(
    labels,
    predictions
)


# ============================================================
# Print results
# ============================================================

print("\n========== CROSS-AGE VERIFICATION RESULTS ==========\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(
    f"\nVerification threshold: {best_threshold:.2f}"
)


# ============================================================
# Save results
# ============================================================

results = {
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "threshold"
    ],

    "value": [
        accuracy,
        precision,
        recall,
        f1,
        auc,
        best_threshold
    ]
}

results_df = pd.DataFrame(results)

results_path = os.path.join(
    RESULTS_DIR,
    "cross_age_verification_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print("\nResults saved to:")
print(results_path)

print("\n========== CROSS-AGE VERIFICATION COMPLETE ==========\n")