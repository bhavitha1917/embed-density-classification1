<<<<<<< HEAD
# Breast Density Classification using Deep Learning and SVM

## Overview

This project focuses on automated mammographic breast density classification using the EMBED dataset.

The objective is to classify mammograms into four density categories:

| Density Label | Description                        |
| ------------- | ---------------------------------- |
| 1             | Almost entirely fatty              |
| 2             | Scattered fibroglandular densities |
| 3             | Heterogeneously dense              |
| 4             | Extremely dense                    |

---

## Dataset

### Training Dataset

Balanced subset selected from EMBED.

* Density 1: 2000 images
* Density 2: 2000 images
* Density 3: 2000 images
* Density 4: 2000 images

Total:

8000 images

### External Test Dataset

* Density 1: 125 images
* Density 2: 125 images
* Density 3: 125 images
* Density 4: 125 images

Total:

500 images

The external test dataset contains images not used during training.

---

## Preprocessing

1. DICOM to PNG conversion
2. Pixel intensity normalization
3. Resize to 1024×1024
4. CLAHE enhancement experiments
5. Data augmentation for deep learning

---

## Methods Evaluated

### Method 1

ResNet50 Feature Extraction + SVM

Pipeline:

ResNet50 → 2048 Features → SVM

---

### Method 2

PCA + SVM

Pipeline:

ResNet50 → PCA (300 Components) → SVM

---

### Method 3

CLAHE + PCA + SVM

Pipeline:

CLAHE → ResNet50 → PCA → SVM

---

### Method 4

Fine-Tuned ResNet50

Trainable Layers:

* Layer3
* Layer4

Classifier:

2048 → 512 → ReLU → Dropout → 4 Classes

---

## Best Results

### Fine-Tuned ResNet50

External Accuracy:

78.0%

Validation Accuracy:

78.12%

---

## Repository Structure

```text
src/
create_labels.py
feature_extraction_resnet50.py
train_svm.py
train_pca_svm.py
external_test_svm.py
clahe_preprocessing.py
finetune_resnet50.py
```

---

## Future Work

* ROI segmentation
* Additional augmentation
* Full EMBED training
=======
# embed-density-classification1
>>>>>>> 2c00c48ad25c0da140d6d4a910ae804600651d91
