import os
import random
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VAL_RATIO = 0.15  # Ratio of training data to use for validation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to the local classification folder
base_dir = './dataset'  # Adjust the path to your local classification folder

# Transforms (Updated with ImageNet normalization)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
}


# Train/Test Split Function
def train_test_split(dataset, test_ratio=0.2, val_ratio=0.15):
    """
    Splits the dataset into train, validation, and test sets based on the ratios provided.
    """
    dataset_size = len(dataset)
    test_size = int(dataset_size * test_ratio)
    val_size = int((dataset_size - test_size) * val_ratio)
    train_size = dataset_size - test_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


# Load and Split Dataset
def load_datasets(base_dir, transforms, test_ratio=0.2, val_ratio=0.15):
    full_dataset = datasets.ImageFolder(base_dir, transform=transforms['train'])
    train_dataset, val_dataset, test_dataset = train_test_split(full_dataset, test_ratio, val_ratio)

    # Apply appropriate transforms to validation and test datasets
    val_dataset.dataset.transform = transforms['val_test']
    test_dataset.dataset.transform = transforms['val_test']

    return train_dataset, val_dataset, test_dataset, full_dataset.classes


# Load datasets
train_dataset, val_dataset, test_dataset, class_names = load_datasets(base_dir, data_transforms)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Compute class weights for imbalanced data
all_labels = [label for _, label in
              train_dataset.dataset.samples]  # Use the full dataset samples for class distribution
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Load pre-trained Vision Transformer
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=len(class_names))
for name, param in model.named_parameters():
    if 'head' not in name:  # Freeze all layers except the classifier head
        param.requires_grad = False
model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


# Training and Validation
def train_and_validate(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss, correct = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct.double() / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())

        # Validation
        model.eval()
        running_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        # Scheduler step
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history


# Train the model
history = train_and_validate(model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer, scheduler, DEVICE)


# Test the model
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

    test_acc = correct.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")


torch.save(model.state_dict(), './vit_office_equipment_model.pth')
