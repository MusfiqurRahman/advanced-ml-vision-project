import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from CustomCNN import CustomCNN  # Assuming CustomCNN is in a file named CustomCNN.py

# Parameters
IMG_SIZE = 256
NUM_CLASSES = 7
BATCH_SIZE = 64  # Increased batch size for efficient GPU utilization
EPOCHS = 200
LEARNING_RATE = 0.0001
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# Set device to use all available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs")

# Define data transformations (augmentation for training, normalization for both)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Random crop with scaling
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset and split into training and validation sets
dataset = datasets.ImageFolder('./dataset', transform=train_transforms)  # Update path to your dataset
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Update validation dataset with different transforms
val_dataset.dataset.transform = val_transforms

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Initialize the model
model = CustomCNN(num_classes=NUM_CLASSES)

# Wrap the model in DataParallel for multi-GPU training
if num_gpus > 1:
    model = nn.DataParallel(model)

model = model.to(device)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Extract the class labels from the training split
train_labels = [train_dataset.dataset.targets[i] for i in train_dataset.indices]

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert to a tensor for PyTorch
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Pass class weights to CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Dynamic learning rate scheduler
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-4, step_size_up=10, mode='triangular')

# Training loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Step the scheduler based on validation loss
    scheduler.step()

# Save the model
torch.save(model.module.state_dict(), 'custom_v2_7.pth')

# Final evaluation on validation data
model.eval()
with torch.no_grad():
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

final_val_accuracy = 100 * val_correct / val_total
print(f'Final Validation Accuracy: {final_val_accuracy:.2f}%')
