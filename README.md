\# Deepfake Detection and Cross-Age Face Verification Using EfficientNet



\## 1. Project Overview



This project implements a deep learning system for deepfake face detection and cross-age face verification using EfficientNet-B0.



The system contains two modules:



1\. \*\*Deepfake Detection:\*\* Classifies an input face image as real or fake.

2\. \*\*Cross-Age Face Verification:\*\* Compares two face images and estimates whether they belong to the same person, even when appearance may differ with age.



A Streamlit web application provides an interface for uploading images and viewing model predictions.



\## 2. Problem Statement



AI-generated and manipulated facial images create challenges for digital identity verification and media authenticity.



Facial appearance can also change significantly over time, making identity verification across different ages difficult.



This project explores deep learning techniques for detecting fake facial images and comparing facial identity across age variations.



\## 3. Objectives



\- Develop an EfficientNet-B0-based deepfake detection model.

\- Classify face images as real or fake.

\- Implement Siamese neural network-based face verification.

\- Evaluate both models using classification metrics.

\- Develop an interactive Streamlit web application.

\- Present experimental results and identify limitations.



\## 4. Technologies Used



\- Python 3.11

\- TensorFlow and Keras

\- EfficientNet-B0

\- OpenCV

\- NumPy

\- Scikit-learn

\- Pandas

\- Streamlit

\- Git and GitHub



\## 5. System Architecture



\### Module 1: Deepfake Detection



1\. Upload a face image.

2\. Resize the image to 224 × 224 pixels.

3\. Process the image using the trained EfficientNet-B0 model.

4\. Predict whether the image is real or fake.

5\. Display the prediction and confidence score.



\### Module 2: Cross-Age Face Verification



1\. Upload two face images.

2\. Resize both images to 224 × 224 pixels.

3\. Pass both images through a Siamese EfficientNet-based model.

4\. Calculate the distance between their learned representations.

5\. Compare the distance against a verification threshold.

6\. Display the predicted same-person or different-person result.



\## 6. Datasets



\### Deepfake Detection



The deepfake detection model was trained and evaluated using the Real vs Fake Faces dataset.



The dataset was organized into training, validation, and testing subsets, each containing real and fake image classes.



\### Cross-Age Face Verification



The cross-age verification experiments used the AgeDB dataset obtained through Hugging Face.



The dataset contains face images with identity and age-related metadata.



Identity-separated training, validation, and test pairs were created for experimental evaluation.



Dataset files are not included in this GitHub repository. Obtain the datasets from their respective sources and prepare the expected folder structure before training.



\## 7. Model Details



\### Deepfake Detection Model



\- Backbone: EfficientNet-B0

\- Input size: 224 × 224 × 3

\- Transfer learning with ImageNet-pretrained weights

\- Global average pooling

\- Dense classification layers

\- Sigmoid output for binary classification

\- Loss: Binary cross-entropy



\### Cross-Age Verification Model



\- Architecture: Siamese neural network

\- Backbone: EfficientNet-based feature extraction

\- Input: Two face images

\- Distance metric: Euclidean distance between normalized embeddings

\- Training objective: Contrastive loss

\- Decision: Compare the predicted distance with a verification threshold



\## 8. Experimental Results



\### Deepfake Detection



| Metric | Result |

|---|---:|

| Test Accuracy | 82.20% |

| Precision | 80.61% |

| Recall | 84.80% |

| ROC-AUC | 90.57% |



The deepfake detection model achieved 82.20% test accuracy and a ROC-AUC of 90.57% on the evaluated test set.



\### Cross-Age Face Verification



| Metric | Result |

|---|---:|

| Test Accuracy | 55.50% |

| Precision | 59.65% |

| Recall | 34.00% |

| F1-score | 43.31% |

| ROC-AUC | 61.18% |



The cross-age verification model demonstrated limited performance in its current experimental configuration. Further training and model improvements are needed.



These results are experimental and should not be interpreted as evidence of production-level reliability.



\## 9. Project Structure



```text

deepfake-cross-age/

├── app/

│   └── app.py

├── src/

│   ├── train\_deepfake.py

│   ├── cross\_age\_verification.py

│   ├── create\_verification\_pairs.py

│   └── train\_siamese.py

├── data/

│   ├── deepfake/

│   └── verification/

├── models/

├── results/

├── notebooks/

├── .gitignore

└── README.md

