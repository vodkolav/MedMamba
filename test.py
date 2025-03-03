import os
import json
import datetime
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_auc_score, f1_score
matplotlib.use('Agg')  # Use a non-interactive backend
from MedMamba import VSSM
from medmnist import INFO, DermaMNIST
class ConfusionMatrix(object):
    """
    Note that if the displayed images are incomplete, it is a matplotlib version problem.
    This routine uses matplotlib-3.2.1 (windows and ubuntu) to draw normally
    Requires additional installation of the prettytable library
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
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
        sum_TP = np.sum(np.diag(self.matrix))
        acc = sum_TP / np.sum(self.matrix)
        print("The model accuracy is: ", acc)

        overall_precision = overall_recall = overall_specificity = 0

        # Initialize a dictionary to store per-class accuracy
        class_accuracies = {}

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = TP / (TP + FP) if TP + FP != 0 else 0.
            Recall = TP / (TP + FN) if TP + FN != 0 else 0.
            Specificity = TN / (TN + FP) if TN + FP != 0 else 0.

            overall_precision += Precision
            overall_recall += Recall
            overall_specificity += Specificity

            # Calculate individual class accuracy
            class_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0.
            class_accuracies[self.labels[i]] = class_accuracy  # Store in the dict

        # Average metrics
        overall_precision /= self.num_classes
        overall_recall /= self.num_classes
        overall_specificity /= self.num_classes

        # Calculate F1-score
        f1 = f1_score(self.all_labels, self.all_preds, average='macro')

        # Calculate AUC using the scores with respect to the labels
        auc = roc_auc_score(pd.get_dummies(self.all_labels),
                            np.array(self.all_scores), multi_class='ovr')

        print(f"Overall Precision: {overall_precision}")
        print(f"Overall Recall (Sensitivity): {overall_recall}")
        print(f"Overall Specificity: {overall_specificity}")
        print(f"F1-score: {f1}")
        print(f"AUC: {auc}")

        # Store metrics for output
        self.metrics = {
            "Overall Accuracy": acc,
            "Overall Precision": overall_precision,
            "Overall Recall": overall_recall,
            "Overall Specificity": overall_specificity,
            "F1-score": f1,
            "AUC": auc
        }

        # Convert summary DataFrame
        self.summarize_metrics(experiment_directory, class_accuracies)


    def summarize_metrics(self, experiment_directory, class_accuracies):
        metrics_df = pd.DataFrame(self.metrics, index=[0])
        metrics_df.to_csv(os.path.join(experiment_directory,"model_metrics.csv"), index=False)
        print(metrics_df)

        # Save individual class accuracies to JSON file
        class_accuracy_file = os.path.join(experiment_directory, "class_accuracies.json")
        with open(class_accuracy_file, 'w') as json_file:
            json.dump(class_accuracies, json_file)
        print(f"Class accuracies saved to {class_accuracy_file}.")
    def to_dict(self):
        # Convert matrix to a list so it can be JSON serializable
        return {
            "matrix": self.matrix.tolist(),
            "num_classes": self.num_classes,
            "labels": self.labels
        }
    def plot(self, experiment_directory):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45, ha='right', fontsize=10)
        plt.yticks(range(self.num_classes), self.labels, fontsize=10)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion Matrix')
        thresh = self.matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(self.matrix[y, x])
                plt.text(x, y, info, verticalalignment='center', horizontalalignment='center',
                         color="white" if info > thresh else "black",
                         fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_directory,'confusion_matrix.png'))  # Save the plot as an image file
        plt.close()  # Close the plot to free up memory

    def save(self, experiment_directory):
        # Save the confusion matrix as a CSV file
        matrix_df = pd.DataFrame(self.matrix, index=self.labels, columns=self.labels)
        matrix_df.to_csv(os.path.join(experiment_directory, 'confusion_matrix.csv'), index=True)

if __name__ == '__main__':
    # Create configuration dictionary
    config = {
        "experiment_name": "test_of_pre_trained_28",
        "MedMNIST_dataset_name": "DermaMNIST",
        "image_size": 28,
        "model_weight_path": 'pretrain_weights/DermaMNIST/Medmamba.pth', #'Train_runs/Train_lr_change_image_28_2025-03-01_18-19-32/medmamba_t_Net.pth', #'Train_runs/Train_2025-02-22_14-08-32/medmamba_t_Net.pth',
        "transform": {
            "resize": (224, 224),
            "normalize": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5]
            }
        },
        "batch_size": 32,
        "test_net": "medmamba_t"  # Choose the appropriate one
    }

    # Dataset name to class mapping
    dataset_class_mapping = {
        "DermaMNIST": DermaMNIST,
        # Add other dataset mappings as needed
    }

    # Use the mapping to get a reference to the dataset class
    dataset_class = dataset_class_mapping[config["MedMNIST_dataset_name"]]

    experiment_name = config["experiment_name"]
    # Create experiment directory based on current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_directory = f"Test_runs/{experiment_name}_{timestamp}"  # Directory path
    os.makedirs(experiment_directory, exist_ok=True)  # Create the directory

    # Save the configuration file in the experiment directory
    config_file_path = os.path.join(experiment_directory, 'config.json')
    with open(config_file_path, 'w') as json_file:
        json.dump(config, json_file)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "test": transforms.Compose([
            transforms.Resize(config["transform"]["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["transform"]["normalize"]["mean"],
                                 std=config["transform"]["normalize"]["std"])
        ])
    }

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform)

    test_dataset = dataset_class(split="test", download=True, size=config["image_size"],
                                transform=data_transform["test"])

    batch_size = config["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=2)
    labels = [v for k, v in test_dataset.info['label'].items()]
    nc = len(labels)

    if config["test_net"] == "medmamba_t":
        net = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=nc).to(device)
    elif config["test_net"] == "medmamba_s":
        net = VSSM(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=nc).to(device)
    elif config["test_net"] == "medmamba_b":
        net = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=nc).to(device)

    # load pretrain weights
    model_weight_path = config["model_weight_path"]
    assert os.path.exists(model_weight_path), f"cannot find {model_weight_path} file"
    weights = torch.load(model_weight_path, map_location=device)
    net.load_state_dict(weights)
    net.to(device)

    confusion = ConfusionMatrix(num_classes=nc, labels=labels)

    net.eval()
    with torch.no_grad():
        for val_data in tqdm(test_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            scores = torch.softmax(outputs, dim=1)  # Use scores for AUC
            preds = torch.argmax(scores, dim=1)
            confusion.update(preds.to("cpu").numpy(),
                             val_labels.to("cpu").numpy(),
                             scores.to("cpu").numpy())

    confusion.save(experiment_directory)
    confusion.plot(experiment_directory)
    confusion.summary()
