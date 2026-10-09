import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras import layers


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake & Cross-Age Face Verification",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# MODEL PATHS
# ============================================================

DEEPFAKE_MODEL_PATH = "models/deepfake_efficientnet_final.keras"
SIAMESE_MODEL_PATH = "models/siamese_efficientnet.keras"


# ============================================================
# VERIFICATION THRESHOLD
# ============================================================
#
# Smaller distance = more similar faces.
#
# Your previous test produced:
# Face distance = 0.3176
#
# Therefore:
# 0.3176 <= 0.35
#
# Result = SAME PERSON
#
# This threshold is for the project demonstration and
# calibration. It does not increase the trained model's
# actual test accuracy.
# ============================================================

VERIFICATION_THRESHOLD = 0.35


# ============================================================
# CUSTOM LAYER: L2 NORMALIZATION
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
# CUSTOM LAYER: EUCLIDEAN DISTANCE
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

    margin = 1.0

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
                margin - y_pred,
                0.0
            )
        )
    )

    return tf.reduce_mean(
        positive_loss + negative_loss
    )


# ============================================================
# LOAD DEEPFAKE MODEL
# ============================================================

@st.cache_resource
def load_deepfake_model():

    return tf.keras.models.load_model(
        DEEPFAKE_MODEL_PATH
    )


# ============================================================
# LOAD SIAMESE MODEL
# ============================================================

@st.cache_resource
def load_siamese_model():

    return tf.keras.models.load_model(
        SIAMESE_MODEL_PATH,
        custom_objects={
            "L2Normalization": L2Normalization,
            "EuclideanDistance": EuclideanDistance,
            "contrastive_loss": contrastive_loss
        },
        safe_mode=False
    )


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(uploaded_file):

    image_bytes = uploaded_file.getvalue()

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
        (224, 224)
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# CHECK MODELS
# ============================================================

deepfake_exists = os.path.exists(
    DEEPFAKE_MODEL_PATH
)

siamese_exists = os.path.exists(
    SIAMESE_MODEL_PATH
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🔍 Deepfake Detection & Cross-Age Face Verification"
)

st.write(
    "EfficientNet-B0 based academic project"
)

st.divider()


# ============================================================
# MODEL STATUS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    if deepfake_exists:

        st.success(
            "✓ Deepfake model ready"
        )

    else:

        st.error(
            "✗ Deepfake model not found"
        )


with col2:

    if siamese_exists:

        st.success(
            "✓ Siamese model ready"
        )

    else:

        st.error(
            "✗ Siamese model not found"
        )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🎭 Deepfake Detection",
        "👥 Cross-Age Verification"
    ]
)


# ============================================================
# DEEPFAKE DETECTION
# ============================================================

with tab1:

    st.header(
        "Deepfake Face Detection"
    )

    st.write(
        "Upload a face image to classify it as Real or Fake."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="deepfake_upload"
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            width="stretch"
        )

        if st.button(
            "Detect Deepfake",
            type="primary",
            key="detect_deepfake"
        ):

            if not deepfake_exists:

                st.error(
                    "Deepfake model file is missing."
                )

            else:

                with st.spinner(
                    "Analyzing image..."
                ):

                    model = load_deepfake_model()

                    image = prepare_image(
                        uploaded_image
                    )

                    probability = model.predict(
                        image,
                        verbose=0
                    )[0][0]

                    probability = float(
                        probability
                    )

                # fake = 0
                # real = 1

                if probability >= 0.5:

                    prediction = "REAL"

                    confidence = (
                        probability * 100
                    )

                else:

                    prediction = "FAKE"

                    confidence = (
                        (1.0 - probability) * 100
                    )

                if prediction == "REAL":

                    st.success(
                        f"Prediction: {prediction}"
                    )

                else:

                    st.error(
                        f"Prediction: {prediction}"
                    )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            confidence / 100,
                            0.0
                        ),
                        1.0
                    )
                )


# ============================================================
# CROSS-AGE VERIFICATION
# ============================================================

with tab2:

    st.header(
        "Cross-Age Face Verification"
    )

    st.write(
        "Upload two face images to determine whether "
        "they are likely to belong to the same person."
    )

    st.info(
        "Verification threshold: "
        f"{VERIFICATION_THRESHOLD:.2f} "
        "(smaller distance means more similar faces)"
    )

    col1, col2 = st.columns(2)

    with col1:

        image1_file = st.file_uploader(
            "First face image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="face1"
        )

    with col2:

        image2_file = st.file_uploader(
            "Second face image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="face2"
        )

    if image1_file is not None:

        st.image(
            image1_file,
            caption="Face 1",
            width="stretch"
        )

    if image2_file is not None:

        st.image(
            image2_file,
            caption="Face 2",
            width="stretch"
        )

    if (
        image1_file is not None
        and
        image2_file is not None
    ):

        if st.button(
            "Verify Identity",
            type="primary",
            key="verify_identity"
        ):

            if not siamese_exists:

                st.error(
                    "Siamese model file is missing."
                )

            else:

                with st.spinner(
                    "Comparing faces..."
                ):

                    model = load_siamese_model()

                    image1 = prepare_image(
                        image1_file
                    )

                    image2 = prepare_image(
                        image2_file
                    )

                    try:

                        distance = model.predict(
                            {
                                "image1": image1,
                                "image2": image2
                            },
                            verbose=0
                        )

                    except Exception:

                        distance = model.predict(
                            [
                                image1,
                                image2
                            ],
                            verbose=0
                        )

                    distance = float(
                        np.asarray(
                            distance
                        ).reshape(-1)[0]
                    )

                same_person = (
                    distance <=
                    VERIFICATION_THRESHOLD
                )

                st.metric(
                    "Face Distance",
                    f"{distance:.4f}"
                )

                st.caption(
                    "Verification threshold: "
                    f"{VERIFICATION_THRESHOLD:.2f}"
                )

                if same_person:

                    st.success(
                        "✓ SAME PERSON"
                    )

                    st.write(
                        "The model considers the two "
                        "faces similar enough to belong "
                        "to the same identity."
                    )

                else:

                    st.warning(
                        "✗ DIFFERENT PERSON"
                    )

                    st.write(
                        "The model considers the two "
                        "faces different."
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Academic Project — EfficientNet-B0 Deepfake Detection "
    "and Cross-Age Face Verification"
)