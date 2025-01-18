# scripts/train.py
import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import CNN

# Load datasets
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
trainset = torch.load(os.path.join(data_dir, "trainset.pth"))
valset = torch.load(os.path.join(data_dir, "valset.pth"))

# Ensure checkpoints directory exists
checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints"))
os.makedirs(checkpoints_dir, exist_ok=True)

# Hyperparameters
learning_rate = 0.005  # Define learning rate
num_epochs = 50
batch_size = 512

# DataLoader function
def get_dataloader(dataset, batch_size=batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

if __name__ == "__main__":
    trainloader = get_dataloader(trainset, shuffle=True)
    valloader = get_dataloader(valset, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs")))

    print(f"Training started with Learning Rate: {learning_rate}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        train_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} - LR: {learning_rate}")

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total

        # **Log Learning Rate & Metrics in TensorBoard**
        writer.add_scalar("Learning Rate", learning_rate, epoch)
        writer.add_scalar("Loss/train", running_loss / len(trainloader), epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Accuracy: {train_accuracy:.2f}% | Learning Rate: {learning_rate}")

        # **Save model checkpoint**
        checkpoint_path = os.path.join(checkpoints_dir, f"cnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # **Verify checkpoint saved successfully**
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint saved: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint NOT saved at {checkpoint_path}")

    writer.close()
    print("Training completed.")
