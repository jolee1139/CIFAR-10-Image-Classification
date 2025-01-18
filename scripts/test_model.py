# scripts/test_model.py
import torch
import random
import os
import matplotlib.pyplot as plt
from model import CNN

# Define CIFAR-10 class labels
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Get absolute path for test dataset
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
testset_path = os.path.join(data_dir, "testset.pth")

# Ensure file exists before loading
if not os.path.exists(testset_path):
    raise FileNotFoundError(f"Dataset file not found: {testset_path}")

# Load test dataset safely
testset = torch.load(testset_path, map_location=torch.device("cpu"))

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = CNN().to(device)
checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/cnn_epoch_50.pth"))

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Function to test a random image
def test_random_image(dataset, model, device):
    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]

    # Prepare image for model input
    image_tensor = image.unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    predicted_label = predicted.item()
    actual_label = label

    # Display the image
    image_to_display = image.permute(1, 2, 0).cpu() * 0.5 + 0.5  # Unnormalize
    plt.imshow(image_to_display)
    plt.title(f"True: {classes[actual_label]}, Predicted: {classes[predicted_label]}")
    plt.axis("off")
    plt.show()

    print(f"True Label: {classes[actual_label]}, Predicted Label: {classes[predicted_label]}")

# Interactive testing loop
def interactive_testing(dataset, model, device):
    while True:
        test_random_image(dataset, model, device)
        user_input = input("Test another image? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting interactive testing.")
            break

# Run interactive testing
interactive_testing(testset, model, device)
