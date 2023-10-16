import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataloader import train_loader, val_loader, test_loader
from model_pytorch import model as pytorch_model  # Use a different variable name

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Load your PyTorch model from model_pytorch module
model =models.resnet18(pretrained = False)  # Use the model loaded from your module
#use PRETRAINED = True for pretrained weights (results in higher accuracy)

#model= pytorch_model 
#use this fr custom written model

# Move the PyTorch model to the GPU if available
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    start_time = time.time()
    print("TRAINING.......")
    model.train()  # Set the model to training mode
    running_loss = 0.0
    i=0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()
        print(running_loss) 
    end_time = time.time()
    epoch_time = end_time - start_time
    remaining_time = (num_epochs - (epoch + 1)) * epoch_time
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f} sec - Estimated Time Remaining: {remaining_time:.2f} sec')
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

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {(100 * correct / total):.2f}%")
