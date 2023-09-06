from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Train and test data directories
train_data_dir = "E:\\try2\\dataset\\archive\\train"
test_data_dir = "E:\\try2\\dataset\\archive\\test"
val_data_dir = "E:\\try2\\dataset\\archive\\valid"

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the train and test datasets using ImageFolder
train_dataset = ImageFolder(train_data_dir, transform=data_transform)
test_dataset = ImageFolder(test_data_dir, transform=data_transform)
val_dataset = ImageFolder(val_data_dir, transform=data_transform)

# Create data loaders for train, validation, and test datasets
batch_size = 64  # Adjust this based on your needs
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=1)  # Batch size 1 for testing



