import torch
import torch.nn as nn
from CustomCNN import CustomCNN
NUM_CLASSES = 7
# Initialize the model
model = CustomCNN(num_classes=NUM_CLASSES)

# Wrap the model in DataParallel for multi-GPU training
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
state_dict = torch.load("custom_v1_7.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
torch.save(model.module.state_dict(), 'custom_v2_7.pth')
