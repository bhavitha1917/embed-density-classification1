
# Breast Density Classification using Deep Learning

## Overview

This project focuses on automated breast density classification from mammography images using deep learning and machine learning techniques. Breast density is an important clinical factor in breast cancer screening and is commonly categorized into four classes corresponding to BI-RADS density categories.

The objective is to classify a mammogram image into one of four density classes:

* Class 1 (A)
* Class 2 (B)
* Class 3 (C)
* Class 4 (D)

---

## Dataset

### Training Dataset

* Dataset: EMBED Mammography Dataset
* Total Images: 8000
* Classes: 4
* Images per Class: 2000
* Image Resolution: 1024 × 1024

### External Test Dataset

* Total Images: 500
* Images per Class: 125
* Used for independent evaluation

---

## Preprocessing

The following preprocessing steps were applied:

1. DICOM to PNG conversion
2. Pixel intensity normalization
3. Image resizing to 1024 × 1024
4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Conversion to RGB format for ResNet50 input

---

## Methods Explored

### 1. ResNet50 Feature Extraction + SVM

Features were extracted using a pretrained ResNet50 model. The final classification layer was removed and the 2048-dimensional feature vectors were used to train an SVM classifier.

### 2. PCA + SVM

Principal Component Analysis (PCA) was applied to reduce the 2048-dimensional feature vectors before SVM classification.

### 3. CLAHE + PCA + SVM

CLAHE preprocessing was applied before feature extraction and PCA-based classification.

### 4. Fine-Tuned ResNet50

A pretrained ResNet50 model was fine-tuned on mammography images using transfer learning. The final classification layer was replaced with a custom classifier for four density classes.

---

## Best Model

Fine-Tuned ResNet50

Training Configuration:

* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Epochs: 7
* Transfer Learning: Last ResNet50 block fine-tuned
* Number of Classes: 4

---

## Results

### Internal Validation

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 77.00% |

### External Test Set

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 79.20% |
| Macro F1 Score    | 79.45% |
| Weighted F1 Score | 79.45% |
| Balanced Accuracy | 79.20% |

---

## Repository Structure

```text
embed-density-classification/

├── checkpoints/
│   └── best_model.pt

├── src/
│   ├── create_labels.py
│   ├── clahe_preprocessing.py
│   ├── feature_extraction_resnet50.py
│   ├── train_svm.py
│   ├── train_pca_svm.py
│   └── finetune_resnet50.py

├── predict.py
├── requirements.txt
├── README.md
└── results.md
```

---

## Usage

Run inference using:

```bash
python predict.py
```

The script loads the trained checkpoint and predicts the breast density class for a given mammogram image.

---

## Future Work

* Fine-tune additional ResNet50 layers
* Advanced augmentation techniques
* Ensemble learning approaches
* Evaluation on larger external datasets

---

## Author

Bhavitha Borra
Tanuh-BCD
