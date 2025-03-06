import datetime
import os
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import auc
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from MedMamba_rejection_learning_entropy import VSSM as medmamba  # import model
from medmnist import INFO, DermaMNIST

class ConfusionMatrix(object):
    """
    Note that if the displayed images are incomplete, it is a matplotlib version problem.
    This routine uses matplotlib-3.2.1 (windows and ubuntu) to draw normally
    Requires additional installation of the prettytable library
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))  # extra class for rejection
        self.num_classes = num_classes + 1  # Increment to include rejection class
        self.labels = labels + [num_classes + 1]  # Add "Rejected" to the labels list
        self.all_preds = []
        self.all_labels = []
        self.all_scores = []

    def update(self, preds, labels, scores):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1
        self.all_preds.extend(preds.flatten())
        self.all_labels.extend(labels.flatten())
        self.all_scores.extend(scores)

    def summary(self):
        # Calculate true positives excluding the rejected class
        sum_TP = np.sum(np.diag(self.matrix[:-1, :-1]))
        total_predictions = np.sum(self.matrix[:-1, :-1])
        # Calculate overall accuracy without the rejected class
        acc = sum_TP / total_predictions if total_predictions > 0 else 0  # Check to avoid division by zero

        return acc

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print(os.getcwd())
    # Create configuration dictionary
    config = dict(experiment_name="Train_Entropy_rejection_Asympt_parameter_Hyperparameter_0.00005_Ridge_alfa_1",
                  MedMNIST_dataset_name="DermaMNIST",
                  train_net="medmamba_t",
                  image_size= 224,
                  model_weight_path ='Train_runs/Train_2025-02-22_14-08-32/medmamba_t_Net.pth',
                  transform= dict(RandomResizedCrop=(224),
                                  resize=(224, 224),
                                  normalize={"mean": [0.5, 0.5, 0.5],
                                              "std": [0.5, 0.5, 0.5]}),
                  batch_size=128,
                  learning_rate=0.01,
                  epochs=10,
                  b_init=2,
                  Hyperparameter=0.00005,
                  Ridge_alfa=1
                  )

    # Dataset name to class mapping
    dataset_class_mapping = dict(DermaMNIST=DermaMNIST)

    # Use the mapping to get a reference to the dataset class
    dataset_class = dataset_class_mapping[config["MedMNIST_dataset_name"]]

    # Create experiment directory based on current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = config["experiment_name"]
    experiment_directory = f"Train_runs/{exp_name}_{timestamp}"
    os.makedirs(experiment_directory, exist_ok=True)  # Create the directory

    # Save the configuration file in the experiment directory
    config_file_path = os.path.join(experiment_directory, 'config.json')
    with open(config_file_path, 'w') as json_file:
        json.dump(config, json_file)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((config["transform"]["resize"])), #([transforms.RandomResizedCrop(config["transform"]["RandomResizedCrop"]),
                                     #transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=config["transform"]["normalize"]["mean"],
                                 std=config["transform"]["normalize"]["std"])]),
        "val": transforms.Compose([transforms.Resize((config["transform"]["resize"])),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=config["transform"]["normalize"]["mean"],
                                 std=config["transform"]["normalize"]["std"])])}

    #model_name = "medMambaOnDermaMnist"
    data_flag = 'dermamnist'

    info = INFO[data_flag]
    task = info['task']
    num_channels = info['n_channels']
    num_classes = len(info['label'])

    train_dataset = dataset_class(split="train", download=True, size=config["image_size"],
                               transform=data_transform["train"])
    train_num = len(train_dataset)

    label_names = [v for k, v in train_dataset.info['label'].items()]
    nc = len(label_names)

    batch_size = config["batch_size"]
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = dataset_class(split="val", download=True, size=config["image_size"],
                                  transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    confidence = 100
    if config["train_net"] == "medmamba_t":
        net = medmamba(b=config["b_init"], confidence=confidence, depths=[2, 2, 4, 2],
                       dims=[96, 192, 384, 768], num_classes=nc)
    elif config["train_net"] == "medmamba_s":
        net = medmamba(b=config["b_init"], confidence=confidence, depths=[2, 2, 8, 2],
                       dims=[96, 192, 384, 768], num_classes=nc)
    elif config["train_net"] == "medmamba_b":
        net = medmamba(b=config["b_init"], confidence=confidence, depths=[2, 2, 12, 2],
                       dims=[128, 256, 512, 1024], num_classes=nc)

    # load pretrain weights
    model_weight_path = config["model_weight_path"]
    assert os.path.exists(model_weight_path), f"cannot find {model_weight_path} file"
    weights = torch.load(model_weight_path, map_location=device)

    # Load existing state dict and handle missing keys
    model_dict = net.state_dict()  # Get the state dict of the model
    pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}  # Only load existing weights
    model_dict.update(pretrained_dict)  # Update model dict with pretrained values

    # Load the updated state dict into the model
    net.load_state_dict(model_dict, strict=False)

    # Freeze all parameters initially
    for param in net.parameters():
        param.requires_grad = False

    # Unfreeze only the parameter associated with b
    net.b.requires_grad = True  # Ensure b is trainable

    confidence_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    net.to(device)

    optimizer = optim.SGD([net.b], lr=config["learning_rate"], weight_decay=config['Ridge_alfa'])

    # Track best AUC
    best_auc = 0
    best_b = net.b.item()
    best_val_auc = 0
    best_val_b = net.b.item()


    rejection_count = 0
    save_path = os.path.join(experiment_directory, f'{config["train_net"]}_Net.pth')
    training_metrics = {
        "epoch": [],
        "b": [],
        "AUC": [],
        "rejection_percentages": [],
        "overall_accuracies": []
    }
    val_metrics = {
        "epoch": [],
        "b": [],
        "AUC": [],
        "rejection_percentages": [],
        "overall_accuracies": []
    }
    auc_history = {
        "epoch": [],
        "b": [],
        "Average AUC": [],
        "Loss": []
    }
    val_auc_history = {
        "epoch": [],
        "b": [],
        "Average AUC": [],
        "Loss": []
    }
    for epoch in range(config["epochs"]):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        # List to accumulate AUC results for the current epoch
        epoch_auc_values = {"AUC": []}  # Store AUC values for the current epoch
        val_epoch_auc_values = {"AUC": []}  # Store AUC values for the current epoch

        for data in enumerate(train_bar):
            images, labels = data[1]
            images = images.to(device)
            # Check if labels are tensors and handle accordingly
            if isinstance(labels, torch.Tensor):
                labels = labels.clone().detach().cpu().numpy()  # Process as tensor
            else:
                labels = np.array(labels)  # Ensure labels are in NumPy format regardless
            optimizer.zero_grad()
            results = {}
            # Forward pass
            for confidence in confidence_levels:
                softmax_probs, preds, should_reject = net(images, confidence=confidence)
                preds = preds.cpu().numpy()
                should_reject = should_reject.cpu().numpy()
                softmax_probs = softmax_probs.cpu().numpy()
                # Initialize Confusion Matrix
                confusion = ConfusionMatrix(num_classes=nc, labels=labels)
                # Handle rejection
                preds[should_reject] = confusion.num_classes - 1  # Assign rejection class index (last index)
                # Update confusion matrix
                confusion.update(preds, labels, softmax_probs)
                # Track rejection statistics
                rejection_count = should_reject.sum().item()
                #print(f'rejection count: {rejection_count}')
                # Calculate percentage of rejections
                rejection_percentage = (rejection_count / len(labels)) * 100
                #print(f'rejection_percentage: {rejection_percentage}')
                #print(f'length of labels: {len(labels)}')
                # Calculate overall accuracy excluding rejections
                valid_labels = labels[preds != confusion.num_classes - 1]
                #valid_labels = valid_labels.squeeze()  # Flatten the labels to remove extra dimension
                #print(f'valid labels: {valid_labels}')
                #print(
                #    f"Type: {type(valid_labels)}, Shape: {valid_labels.shape if hasattr(valid_labels, 'shape') else 'No shape attribute'}")
                valid_labels = torch.tensor(valid_labels).reshape(-1)
                # Calculate overall accuracy excluding rejections
                #print(
                #    f"Type: {type(valid_labels)}, Shape: {valid_labels.shape if hasattr(valid_labels, 'shape') else 'No shape attribute'}")
                if len(valid_labels) > 0:
                    overall_accuracy_non_rejected = confusion.summary()
                    #print(f'overall accuracy non rejected: {overall_accuracy_non_rejected}')
                else:
                    overall_accuracy_non_rejected = 0

                # Store results for plotting
                results[confidence] = {
                    "rejection_percentage": rejection_percentage,
                    "overall_accuracy": overall_accuracy_non_rejected
                }

            # Prepare data for auc calculation from results
            rejection_percentages = [results[confidence]['rejection_percentage'] for confidence in
                                    confidence_levels]
            overall_accuracies = [results[confidence]['overall_accuracy'] for confidence in
                                    confidence_levels]

            # Combine rejection percentages and overall accuracies into a list of tuples
            combined_metrics = list(zip(rejection_percentages, overall_accuracies))

            # Sort based on rejection percentages
            sorted_metrics = sorted(combined_metrics, key=lambda x: x[0])

            # Unzip sorted metrics back to two lists
            sorted_rejection_percentages, sorted_overall_accuracies = zip(*sorted_metrics)

            # Calculate AUC
            AUC = auc(sorted_rejection_percentages, sorted_overall_accuracies)
            epoch_auc_values['AUC'].append(AUC)
            current_b_value = net.b.item() # Get the current value of b
            #print(f' AUC: {AUC} for current_b_value: {current_b_value}')
            training_metrics["epoch"].append(epoch)
            training_metrics["b"].append(current_b_value)  # Track the current b value
            training_metrics["AUC"].append(AUC)  # Append the last calculated AUC to metrics
            training_metrics["rejection_percentages"].append(rejection_percentages)
            training_metrics["overall_accuracies"].append(overall_accuracies)

            # Compute evaluation metrics
            # Compute a pseudo-loss indicating AUC
            auc_tensor = torch.tensor(AUC, requires_grad=False)  # Make auc a tensor
            # Implement the reward/loss function considering AUC as x and net.b as y
            x = auc_tensor
            y = net.b
            L = config["Hyperparameter"]
            E = 0.0000001
            f = 1/(E + x) + (L * y)

            # Minimize the negative of this function to maximize f
            loss = f
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} AUC:{:.3f} for current b value:{} Gradient of b:{}".format(epoch + 1,
                                                                     config["epochs"],
                                                                     running_loss,
                                                                     AUC,
                                                                     current_b_value,
                                                                     net.b.grad)
        # Calculate average AUC for the current epoch
        auc_per_epoch = np.mean(epoch_auc_values["AUC"]) if len(epoch_auc_values["AUC"]) > 0 else 0
        #print(f'epoch_auc_values: {epoch_auc_values}')
        #print(f'auc_per_epoch: {auc_per_epoch}')
        auc_history["epoch"].append(epoch)
        auc_history["b"].append(net.b.item())
        auc_history["Average AUC"].append(auc_per_epoch)
        auc_history["Loss"].append(loss.item())

        # Keep track of the best `b` based on AUC
        if auc_per_epoch > best_auc:
            best_auc = auc_per_epoch
            best_b = net.b.item()

        print(f'Epoch {epoch +1}, Current AUC: {auc_per_epoch:.3f}, Best AUC: {best_auc:.3f} with b: {best_b:.3f}')



        # validate
        net.eval()
        val_AUC = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                # Check if val_labels are tensors and handle accordingly
                if isinstance(val_labels, torch.Tensor):
                    val_labels = val_labels.clone().detach().cpu().numpy()  # Process as tensor
                else:
                    val_labels = np.array(val_labels)  # Ensure val_labels are in NumPy format regardless
                val_results = {}
                for confidence in confidence_levels:
                    val_softmax_probs, val_preds, val_should_reject = net(val_images, confidence=confidence)
                    val_preds = val_preds.cpu().numpy()
                    val_should_reject = val_should_reject.cpu().numpy()
                    val_softmax_probs = val_softmax_probs.cpu().numpy()
                    # Initialize Confusion Matrix
                    confusion = ConfusionMatrix(num_classes=nc, labels=val_labels)
                    # Handle rejection
                    val_preds[val_should_reject] = confusion.num_classes - 1  # Assign rejection class index (last index)
                    # Update confusion matrix
                    confusion.update(val_preds, val_labels, val_softmax_probs)
                    # Track rejection statistics
                    val_rejection_count = val_should_reject.sum().item()
                    # Calculate percentage of rejections
                    val_rejection_percentage = (val_rejection_count / len(val_labels)) * 100
                    # Calculate overall accuracy excluding rejections
                    val_valid_labels = val_labels[val_preds != confusion.num_classes - 1]
                    #val_valid_labels = val_valid_labels.squeeze()  # Flatten the labels to remove extra dimension
                    # Ensure it's an iterable, even if it results in an empty collection
                    val_valid_labels = torch.tensor(val_valid_labels).reshape(-1)

                    # Calculate overall accuracy excluding rejections
                    if len(val_valid_labels) > 0:
                        val_overall_accuracy_non_rejected = confusion.summary()
                    else:
                        val_overall_accuracy_non_rejected = 0

                    # Store results for plotting
                    val_results[confidence] = {
                        "rejection_percentage": val_rejection_percentage,
                        "overall_accuracy": val_overall_accuracy_non_rejected
                    }

                # Prepare data for auc calculation from results
                val_rejection_percentages = [val_results[confidence]['rejection_percentage'] for confidence in
                                         confidence_levels]
                val_overall_accuracies = [val_results[confidence]['overall_accuracy'] for confidence in
                                      confidence_levels]

                # Combine rejection percentages and overall accuracies into a list of tuples
                combined_val_metrics = list(zip(val_rejection_percentages, val_overall_accuracies))

                # Sort based on rejection percentages
                sorted_val_metrics = sorted(combined_val_metrics, key=lambda x: x[0])

                # Unzip sorted metrics back to two lists
                sorted_val_rejection_percentages, sorted_val_overall_accuracies = zip(*sorted_val_metrics)

                # Calculate AUC
                val_AUC = auc(sorted_val_rejection_percentages, sorted_val_overall_accuracies)
                val_epoch_auc_values['AUC'].append(val_AUC)
                current_b_value = net.b.item()  # Get the current value of b
                val_metrics["epoch"].append(epoch)
                val_metrics["b"].append(current_b_value)  # Track the current b value
                val_metrics["AUC"].append(val_AUC)  # Append the last calculated AUC to metrics
                val_metrics["rejection_percentages"].append(val_rejection_percentages)
                val_metrics["overall_accuracies"].append(val_overall_accuracies)
        # Calculate average AUC for the current epoch
        val_auc_per_epoch = np.mean(val_epoch_auc_values["AUC"]) if len(val_epoch_auc_values["AUC"]) > 0 else 0
        #print(f'val_epoch_auc_values: {val_epoch_auc_values}')
        #print(f'val_auc_per_epoch: {val_auc_per_epoch}')
        val_auc_history["epoch"].append(epoch)
        val_auc_history["b"].append(net.b.item())
        val_auc_history["Average AUC"].append(val_auc_per_epoch)
        val_auc_history["Loss"].append(loss.item())

        if val_auc_per_epoch > best_val_auc:
            best_val_auc = val_auc_per_epoch
            best_val_b = net.b.item()
            torch.save(net.state_dict(), save_path)

        print(f'Epoch {epoch + 1}, Current validation AUC: {val_auc_per_epoch:.3f}, Best validation AUC: {best_val_auc:.3f} with b: {best_val_b:.3f}')

        # Plot loss and accuracy after each epoch
        plt.figure(figsize=(18, 5))

        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(auc_history["Loss"], label='Train Loss', color='tab:red')
        plt.title('Train Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(val_auc_history["Average AUC"], label='Mean Validation AUC', color='tab:blue')
        plt.title('Mean Validation AUC Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Validation AUC')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 3, 3)
        plt.plot(val_auc_history["b"], label='b Parameter', color='tab:green')
        plt.title('b Parameter Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('b Parameter')
        plt.legend()

        # Save the plot as an image file in the experiment directory
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_directory, f'Mean_AUC_epoch.png'))
        plt.close()  # Close plot to free memory

        full_train_metrics_df = pd.DataFrame(training_metrics)
        full_train_metrics_df.to_csv(os.path.join(experiment_directory, 'full_training_auc_df.csv'), index=False)
        full_val_metrics_df = pd.DataFrame(val_metrics)
        full_val_metrics_df.to_csv(os.path.join(experiment_directory, 'full_validation_auc_df.csv'), index=False)
        training_auc_df = pd.DataFrame(auc_history)
        training_auc_df.to_csv(os.path.join(experiment_directory, 'training_auc_df.csv'), index=False)
        val_auc_df = pd.DataFrame(val_auc_history)
        val_auc_df.to_csv(os.path.join(experiment_directory, 'validation_auc_df.csv'), index=False)

    print(f'Finished Training.')
    print(f'Best Training AUC: {best_auc:.3f} with b: {best_b:.3f}')
    print(f'Best validation AUC: {best_val_auc:.3f} with b: {best_val_b:.3f}')

if __name__ == '__main__':
    main()
