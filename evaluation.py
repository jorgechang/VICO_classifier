"""Evaluation of CNN"""
# Standar modules
import argparse

# Third party modules
import matplotlib.pyplot as plt
import torch
from plot_metric.functions import BinaryClassification
from torchvision import datasets, transforms

# Local modules
from utils.constants import SIZE, MEAN, STD


# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="model_best.pth.tar",
    help="path to input PyTorch model (default: model_best.pth.tar)",
)
parser.add_argument("--test", type=str, default="", help="desired path of images")
parser.add_argument("--type", type=str, default="", help="age, gender, or employee")
opt = parser.parse_args()

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Type of dataset
dataset_type = opt.type
if dataset_type == "gender":
    LABELS_MAP = ['f','m']
elif dataset_type == "employee":
    LABELS_MAP = ['client','employee']
else:
    LABELS_MAP = ['0-15','16-24','25-34','35-44','45-54','55-65','65-100']

# Load model
model = torch.load(opt.model)
model.to(device)
model.eval()

# Define image transformations
tfms = transforms.Compose(
    [
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

# Create Dataset and Dataloader from folder
dataset = datasets.ImageFolder(opt.test, transform=tfms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

# Performs forward pass for classification
for batch, labels in iter(dataloader):
    batch, labels = batch.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(batch)
        outputs = torch.softmax(outputs, dim=1).argmax(dim=1)


# Visualisation with plot_metric [https://github.com/yohann84L/plot_metric]
bc = BinaryClassification(
    labels.cpu().data.numpy(), outputs.cpu().data.numpy(), labels=LABELS_MAP
)

# Figures
plt.figure(figsize=(15, 10))
plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
bc.plot_roc_curve()
plt.subplot2grid((2, 6), (0, 2), colspan=2)
bc.plot_precision_recall_curve()
plt.subplot2grid((2, 6), (0, 4), colspan=2)
bc.plot_class_distribution()
plt.subplot2grid((2, 6), (1, 1), colspan=2)
bc.plot_confusion_matrix()
plt.subplot2grid((2, 6), (1, 3), colspan=2)
bc.plot_confusion_matrix(normalize=True)
plt.show()
bc.print_report()