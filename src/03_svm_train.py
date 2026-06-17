import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading features...")

X = np.load("X.npy")

y = np.load("y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training SVM...")

svm = SVC(
    kernel='rbf',
    C=10
)

svm.fit(
    X_train,
    y_train
)

preds = svm.predict(
    X_test
)

print(
    "\nAccuracy:"
)

print(
    accuracy_score(
        y_test,
        preds
    )
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        preds
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        preds
    )
)