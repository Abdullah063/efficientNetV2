import torch
from torchvision import transforms

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Paths
DATA_DIR = '/Users/altun/Desktop/lostProject/FD'

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10

# Image transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])