import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import train_loader, val_loader
from model_pytorch import model as pytorch_model  # Use a different variable name

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Load your PyTorch model from model_pytorch module
model = pytorch_model  # Use the model loaded from your module

# Move the PyTorch model to the GPU if available
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    print("TRAINING.......")
    model.train()  # Set the model to training mode
    running_loss = 0.0
    i=0
    for inputs, labels in train_loader:
        print("FEEDING...")
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        print(f"FEEDING DONE....{i}")
        running_loss += loss.item()
        print(running_loss)
        i+=1
    # Print training loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Validation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {(100 * correct / total):.2f}%")
