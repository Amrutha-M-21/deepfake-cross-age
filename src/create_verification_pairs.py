import os
import random
import pandas as pd

# ============================================================
# Configuration
# ============================================================

METADATA_PATH = "data/verification/agedb/train/metadata.csv"
IMAGE_ROOT = "data/verification/agedb/train"

OUTPUT_DIR = "data/verification/pairs"

TRAIN_IDENTITIES = 70
VAL_IDENTITIES = 10
TEST_IDENTITIES = 10

POSITIVE_PAIRS_PER_IDENTITY = 10
NEGATIVE_PAIRS_PER_IDENTITY = 10

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load metadata
# ============================================================

print("\n========== LOADING AGEDB ==========\n")

df = pd.read_csv(METADATA_PATH)

df["full_path"] = df["file_name"].apply(
    lambda x: os.path.join(
        IMAGE_ROOT,
        x.replace("/", os.sep)
    )
)

# Keep only existing images
df = df[
    df["full_path"].apply(os.path.exists)
].reset_index(drop=True)

print("Images found:", len(df))
print("Identities found:", df["identity"].nunique())


# ============================================================
# Keep identities with at least 2 different ages
# ============================================================

valid_identities = []

for identity, group in df.groupby("identity"):

    unique_ages = group["age"].nunique()

    if len(group) >= 2 and unique_ages >= 2:
        valid_identities.append(identity)

print(
    "Identities with multiple ages:",
    len(valid_identities)
)


# ============================================================
# Shuffle identities
# ============================================================

random.shuffle(valid_identities)

required = (
    TRAIN_IDENTITIES +
    VAL_IDENTITIES +
    TEST_IDENTITIES
)

if len(valid_identities) < required:
    raise ValueError(
        f"Not enough identities. "
        f"Need {required}, "
        f"but found {len(valid_identities)}."
    )


train_ids = valid_identities[
    :TRAIN_IDENTITIES
]

val_ids = valid_identities[
    TRAIN_IDENTITIES:
    TRAIN_IDENTITIES + VAL_IDENTITIES
]

test_ids = valid_identities[
    TRAIN_IDENTITIES + VAL_IDENTITIES:
    TRAIN_IDENTITIES + VAL_IDENTITIES + TEST_IDENTITIES
]


print("\nIdentity split:")
print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# Pair generation function
# ============================================================

def create_pairs(
    identities,
    positive_per_identity,
    negative_per_identity
):

    positive_pairs = []

    negative_pairs = []

    # --------------------------------------------------------
    # Positive pairs
    # --------------------------------------------------------

    for identity in identities:

        group = df[
            df["identity"] == identity
        ]

        rows = list(
            group.iterrows()
        )

        possible_positive = []

        for i in range(len(rows)):

            for j in range(i + 1, len(rows)):

                _, row1 = rows[i]
                _, row2 = rows[j]

                # Cross-age pair only
                if row1["age"] == row2["age"]:
                    continue

                possible_positive.append(
                    {
                        "image1": row1["full_path"],
                        "image2": row2["full_path"],
                        "identity1": row1["identity"],
                        "identity2": row2["identity"],
                        "age1": row1["age"],
                        "age2": row2["age"],
                        "label": 1
                    }
                )

        random.shuffle(possible_positive)

        positive_pairs.extend(
            possible_positive[
                :positive_per_identity
            ]
        )

    # --------------------------------------------------------
    # Negative pairs
    # --------------------------------------------------------

    identity_list = list(identities)

    for identity in identities:

        group1 = df[
            df["identity"] == identity
        ]

        for _ in range(
            negative_per_identity
        ):

            other_identity = random.choice(
                [
                    x for x in identity_list
                    if x != identity
                ]
            )

            group2 = df[
                df["identity"] == other_identity
            ]

            row1 = group1.sample(
                1,
                random_state=random.randint(
                    0,
                    999999
                )
            ).iloc[0]

            row2 = group2.sample(
                1,
                random_state=random.randint(
                    0,
                    999999
                )
            ).iloc[0]

            negative_pairs.append(
                {
                    "image1": row1["full_path"],
                    "image2": row2["full_path"],
                    "identity1": row1["identity"],
                    "identity2": row2["identity"],
                    "age1": row1["age"],
                    "age2": row2["age"],
                    "label": 0
                }
            )

    pairs = (
        positive_pairs +
        negative_pairs
    )

    random.shuffle(pairs)

    return pd.DataFrame(pairs)


# ============================================================
# Create datasets
# ============================================================

print("\n========== CREATING TRAIN PAIRS ==========\n")

train_pairs = create_pairs(
    train_ids,
    POSITIVE_PAIRS_PER_IDENTITY,
    NEGATIVE_PAIRS_PER_IDENTITY
)

print(
    "Training pairs:",
    len(train_pairs)
)


print("\n========== CREATING VALIDATION PAIRS ==========\n")

val_pairs = create_pairs(
    val_ids,
    POSITIVE_PAIRS_PER_IDENTITY,
    NEGATIVE_PAIRS_PER_IDENTITY
)

print(
    "Validation pairs:",
    len(val_pairs)
)


print("\n========== CREATING TEST PAIRS ==========\n")

test_pairs = create_pairs(
    test_ids,
    POSITIVE_PAIRS_PER_IDENTITY,
    NEGATIVE_PAIRS_PER_IDENTITY
)

print(
    "Test pairs:",
    len(test_pairs)
)


# ============================================================
# Save pairs
# ============================================================

train_path = os.path.join(
    OUTPUT_DIR,
    "train_pairs.csv"
)

val_path = os.path.join(
    OUTPUT_DIR,
    "val_pairs.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test_pairs.csv"
)


train_pairs.to_csv(
    train_path,
    index=False
)

val_pairs.to_csv(
    val_path,
    index=False
)

test_pairs.to_csv(
    test_path,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n========== PAIR CREATION COMPLETE ==========\n")

print(
    "Train pairs:",
    len(train_pairs)
)

print(
    "Validation pairs:",
    len(val_pairs)
)

print(
    "Test pairs:",
    len(test_pairs)
)

print("\nSaved files:")

print(train_path)
print(val_path)
print(test_path)