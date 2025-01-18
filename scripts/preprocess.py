# scripts/preprocess.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# CIFAR-10 Class Labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Define dataset transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(data_dir, exist_ok=True)

# Download CIFAR-10 dataset
print("Downloading CIFAR-10 dataset...")
full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

# Split training set into 80% train, 20% validation
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

# Save datasets
torch.save(trainset, os.path.join(data_dir, "trainset.pth"))
torch.save(valset, os.path.join(data_dir, "valset.pth"))
torch.save(testset, os.path.join(data_dir, "testset.pth"))

# Print dataset sizes
print(f"Train set size: {len(trainset)} images")
print(f"Validation set size: {len(valset)} images")
print(f"Test set size: {len(testset)} images")

# Function to visualize first 10 images
def visualize_images(dataset, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 2))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0) * 0.5 + 0.5  # Unnormalize
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(CLASS_NAMES[label])
    plt.show()

print("Visualizing first 10 images from the training dataset...")
visualize_images(trainset)
