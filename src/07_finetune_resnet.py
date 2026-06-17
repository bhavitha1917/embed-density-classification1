import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

import time

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(device)

#datasets from kaggle

train_dir = "/kaggle/input/datasets/borrabhavitha/images-clahe/Test_images_clahe/Test_images_clahe/train"

val_dir = "/kaggle/input/datasets/borrabhavitha/images-clahe/Test_images_clahe/Test_images_clahe/val"

test_dir = "/kaggle/input/datasets/borrabhavitha/images-clahe/extest_clahe/extest_clahe"

#augmentations

transform_train = transforms.Compose([

    transforms.Resize((384,384)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor()
])

transform_val = transforms.Compose([

    transforms.Resize((384,384)),

    transforms.ToTensor()
])

#datasets

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=transform_val
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform_val
)

#dataloaders

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

#model

model = models.resnet50(
    weights="DEFAULT"
)

#freeze evrything
for param in model.parameters():

    param.requires_grad = False

#unfreeze 
for param in model.layer3.parameters():

    param.requires_grad = True

for param in model.layer4.parameters():

    param.requires_grad = True

#new classifier

model.fc = nn.Sequential(

    nn.Linear(
        model.fc.in_features,
        512
    ),

    nn.ReLU(),

    nn.Dropout(0.3),

    nn.Linear(
        512,
        4
    )
)

#change to gpu

model = model.to(device)

#loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-4
)

#training loop

best_val_acc = 0

num_epochs = 7

for epoch in range(num_epochs):

    start = time.time()
    
    #training
    
    model.train()

    correct = 0
    total = 0

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
                        labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    #validation

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:
    
            images = images.to(device)
            labels = labels.to(device)
                
            outputs = model(images)
    
            _, predicted = torch.max(outputs, 1)
    
            val_total += labels.size(0)
    
            val_correct += (
                predicted == labels
            ).sum().item()

    val_acc = 100 * val_correct / val_total
    train_acc = 100 * correct / total

    elapsed = (
        time.time() - start
    )

    print(
        f"Epoch {epoch+1}/{num_epochs}"
        f" | Train Acc: {train_acc:.2f}%"
        f" | Val Acc: {val_acc:.2f}%"
        f" | Time: {elapsed:.1f}s"
        )

    if val_acc > best_val_acc:
    
            best_val_acc = val_acc
                
            torch.save(
                model.state_dict(),
                "/kaggle/working/best_density_model.pth"
            )
    
            print("Saved Best Model")

#ext testing

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )
        
acc = accuracy_score(
    all_labels,
    all_preds
)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)

weighted_f1 = f1_score(
    all_labels,
    all_preds,
    average="weighted"
)

balanced_acc = balanced_accuracy_score(
    all_labels,
    all_preds
)

print(f"Accuracy: {acc:.4f}")

print(
    f"Macro F1: {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_acc:.4f}"
) 

