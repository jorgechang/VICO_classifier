"""Functions for the classification"""
# Standar modules
import copy
import time

# Third party modules
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def get_features(name, features):
    """Defines hook to specific layer

    Parameters:
    -----------
        name [String]:
            name of embedding block
        features [Dictionary]:
            Dictionary to save features at layer
    """

    def hook(output):
        features[name] = output.detach()

    return hook

def print_summary(model_ft):
    """Prints model summary information

    Parameters:
       -----------
           model_ft [customVGG]:
               VGG custom model for classification
    """
    print("Model Summary:-\n")
    for num, (name, param) in enumerate(model_ft.named_parameters()):
        print(num, name, param.requires_grad)
    summary(model_ft, input_size=(3, 224, 128))
    print(model_ft)

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    class_names,
    dataset_sizes,
    num_epochs=20,
):
    """Defines model training routine

    Parameters:
    -----------
        model [customVGG]:
            VGG custom model for training
        criterion [torch.nn.CrossEntropyLoss]:
            Pytorch loss function
        optimizer [torch.optim.Optimizer]:
            Optimization algorithm
        scheduler [torch.optim.lr_scheduler]:
            Scheduler to adjust learning rate
        dataloaders [dict]:
            Dictionary with DataLoaders for training and validation
        device [torch.device]:
            Device where model and data will live
        class_names [list]:
            List of labeled classes
        dataset_sizes [dict]:
            Dictionary with lenght of datasets
        num_epochs [int]:
            Number of epochs of training routine

    Returns:
    --------
        model [customVGG]:
            model with best weights
    """

    # Start time  of training
    since = time.time()

    # Initialize varibales that stores best weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter("")

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Metric variables per epoch
            running_loss = 0.0
            running_corrects = 0
            running_topk_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # Perform topK stats
                    if len(class_names) > 2:
                        _, top_k_preds = torch.topk(outputs, 2)
                    loss = criterion(outputs, labels)

                    # backward propagation + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Loss and accuracy metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if len(class_names) > 2:
                    running_topk_corrects += torch.sum(
                        top_k_preds[:, 0] == labels.data
                    ) + torch.sum(top_k_preds[:, 1] == labels.data)

            # Perform scheduler step
            if phase == "train":
                scheduler.step()

            # Update epoch loss and total/topK accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if len(class_names) > 2:
                epoch_acc_k = running_topk_corrects.double() / dataset_sizes[phase]
            else:
                epoch_acc_k = 0.0

            print(
                "{} Loss: {:.4f} Acc: {:.4f} Top2Acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc, epoch_acc_k
                )
            )

            # Record training loss and accuracy for each phase
            if phase == "train":
                writer.add_scalar("Train/Loss", epoch_loss, epoch)
                writer.add_scalar("Train/Accuracy", epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar("Valid/Loss", epoch_loss, epoch)
                writer.add_scalar("Valid/Accuracy", epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Record and print elapsed training time
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Print best accuracy achieved
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
