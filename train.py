"""Train CNN models for age, gender, and employee classification"""
# Standar modules
import argparse
import multiprocessing

# Third party modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

# Local modules
from nets import customVGG
from utils.constants import SIZE, MEAN, STD
from utils.functions import print_summary, train_model

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--data_path", required=True, help="dataset path")
ap.add_argument("--type", required=True, help="Dataset mode: gender/age/employee")
ap.add_argument("--save_dir", help="Save checkpoint directory")
ap.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="images per gpu, the total batch size is $NGPU x batch_size",
)
ap.add_argument(
    "--epochs", default=26, type=int, metavar="N", help="number of total epochs to run"
)
args = vars(ap.parse_args())

# Set the train and validation directory paths
train_directory = args["data_path"] + "/train"
valid_directory = args["data_path"] + "/val"

# Set model save path
PATH = ap["save_dir"]
# Batch size
bs = args["batch_size"]
# Number of epochs
num_epochs = args["epochs"]
# Number of workers
num_cpu = multiprocessing.cpu_count()
# Dataset type
type_mode = args["type"]

# Define data preprocess pipeline
image_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(size=SIZE),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize(size=SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    ),
}

# Load data from folders
dataset = {
    "train": datasets.ImageFolder(
        root=train_directory, transform=image_transforms["train"]
    ),
    "valid": datasets.ImageFolder(
        root=valid_directory, transform=image_transforms["valid"]
    ),
}

# Size of train and validation data
dataset_sizes = {"train": len(dataset["train"]), "valid": len(dataset["valid"])}

# Create iterators for data loading
dataloaders = {
    "train": data.DataLoader(
        dataset["train"],
        batch_size=bs,
        shuffle=True,
        num_workers=num_cpu,
        pin_memory=True,
        drop_last=True,
    ),
    "valid": data.DataLoader(
        dataset["valid"],
        batch_size=bs,
        shuffle=True,
        num_workers=num_cpu,
        pin_memory=True,
        drop_last=True,
    ),
}

# Class names or target labels
class_names = dataset["train"].classes
print("Classes:", class_names)

# Print the train and validation data sizes
print(
    "Training-set size:",
    dataset_sizes["train"],
    "\nValidation-set size:",
    dataset_sizes["valid"],
)

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load custom VGG models
print("\nLoading VGG model for training from scratch ...\n")
if type_mode == "gender":
    model_ft = customVGG(in_ch=3, num_classes=2)
elif type_mode == "age":
    model_ft = customVGG(in_ch=3, num_classes=len(class_names))
elif type_mode == "employee":
    model_ft = customVGG(in_ch=3, num_classes=2)

# Send the model to device
model_ft = model_ft.to(device)

# Print model summary
print_summary(model_ft)

# Define Loss function
criterion = nn.CrossEntropyLoss()

# Define Optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# Define learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# Train the model
model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    dataloaders,
    device,
    class_names,
    dataset_sizes,
    num_epochs=num_epochs,
)
# Save the entire model in path
print("\nSaving the model...")
torch.save(model_ft, PATH)
