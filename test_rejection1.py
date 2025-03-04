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
from sklearn.metrics import roc_auc_score, f1_score, auc

matplotlib.use('Agg')  # Use a non-interactive backend
from MedMamba_rejection1 import VSSM
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
        self.labels = labels + ["Rejected"]  # Add "Rejected" to the labels list
        self.all_preds = []
        self.original_preds = []
        self.all_labels = []
        self.original_labels = []
        self.all_scores = []
        self.original_scores = []


    def update(self, preds, labels, scores):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1
            if p != self.num_classes - 1:  # Check for rejection
                self.original_labels.extend(labels.flatten())
                self.original_preds.extend(preds.flatten())
                self.original_scores.extend(scores)
        self.all_preds.extend(preds.flatten())
        self.all_labels.extend(labels.flatten())
        self.all_scores.extend(scores)

    def summary(self, experiment_directory, confidence):
        # Calculate true positives excluding the rejected class
        sum_TP = np.sum(np.diag(self.matrix[:-1, :-1]))  # Exclude last row and column
        print(f'sum TP: {sum_TP}')
        total_predictions = np.sum(self.matrix[:-1, :-1])  # Total excluding rejected class
        print(f'total predictions: {total_predictions}')
        # Calculate overall accuracy without the rejected class
        acc = sum_TP / total_predictions if total_predictions > 0 else 0  # Check to avoid division by zero
        print(f"The model accuracy (excluding rejected class) for confidence level {confidence} is: ", acc)

        overall_precision = overall_recall = overall_specificity = overall_F1 = 0

        # Initialize a dictionary to store per-class accuracy
        class_accuracies = {}

        # Initialize list to store AUC scores
        auc_scores = []
        print(f'number of classes: {self.num_classes}')
        for i in range(self.num_classes - 1):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :-1]) - TP
            FN = np.sum(self.matrix[:-1, i]) - TP
            TN = np.sum(self.matrix[:-1, :-1]) - TP - FP - FN
            Precision = TP / (TP + FP) if TP + FP != 0 else 0.
            Recall = TP / (TP + FN) if TP + FN != 0 else 0.
            Specificity = TN / (TN + FP) if TN + FP != 0 else 0.
            F1 = (2 * TP) / (2 * (TP + FP + FN)) if TP + FP + FN != 0 else 0.

            overall_precision += Precision
            overall_recall += Recall
            overall_specificity += Specificity
            overall_F1 += F1

            # Calculate individual class accuracy
            class_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0.
            class_accuracies[self.labels[i]] = class_accuracy  # Store in the dict

            # Calculate ROC AUC for binary evaluation per class
            binary_labels = (np.array(self.all_labels) == i).astype(int)  # Treat i as positive class
            binary_scores = np.array(self.all_scores)[:, i]  # Get scores for the current class

            if len(binary_scores) > 0:
                auc_score = roc_auc_score(binary_labels, binary_scores)  # Calculate AUC for this class
                auc_scores.append(auc_score)  # Store the AUC score for averaging


        # Calculate average AUC score from individual class evaluations
        average_auc_score = np.mean(auc_scores) if auc_scores else 0

        # Average metrics
        overall_precision /= (self.num_classes - 1)  # Exclude rejected class
        overall_recall /= (self.num_classes - 1)
        overall_specificity /= (self.num_classes - 1)
        overall_F1 /= (self.num_classes - 1)

        # Calculate F1-score
        f1 = f1_score(self.original_labels, self.original_preds, average='macro')
        f1_score2 = 0
        if (overall_precision + overall_recall) != 0:
            f1_score2 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)
        # Calculate AUC using the scores with respect to the labels
        ROC_AUC_score = 0
        if len(self.original_scores) > 0:
            ROC_AUC_score = roc_auc_score(pd.get_dummies(self.original_labels),
                            np.array(self.original_scores), multi_class='ovr')

        print(f"Overall Precision: {overall_precision}")
        print(f"Overall Recall (Sensitivity): {overall_recall}")
        print(f"Overall Specificity: {overall_specificity}")
        print(f"F1-score: {f1}")
        print(f"F1-score2: {f1_score2}")
        print(f"F1-score3: {overall_F1}")
        print(f"ROC AUC score: {ROC_AUC_score}")
        print(f"Average ROC AUC score (across all classes): {average_auc_score}")

        # Store metrics for output
        self.metrics = {
            "Overall Accuracy": acc,
            "Overall Precision": overall_precision,
            "Overall Recall": overall_recall,
            "Overall Specificity": overall_specificity,
            "F1-score": f1,
            "F1-score2": f1_score2,
            "F1-score3": overall_F1,
            "ROC AUC score": ROC_AUC_score,
            "Average ROC AUC score": average_auc_score
        }

        # Convert summary DataFrame
        self.summarize_metrics(experiment_directory, class_accuracies, confidence)
        return acc
    def summarize_metrics(self, experiment_directory, class_accuracies, confidence):
        metrics_df = pd.DataFrame(self.metrics, index=[0])
        metrics_df.to_csv(os.path.join(experiment_directory, f"model_metrics_for_confidence_{confidence}.csv"),
                          index=False)
        print(metrics_df)

        # Save individual class accuracies to JSON file
        class_accuracy_file = os.path.join(experiment_directory, f"class_accuracies_for_confidence_{confidence}.json")
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

    def plot(self, experiment_directory, confidence):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45, ha='right', fontsize=10)
        plt.yticks(range(self.num_classes), self.labels, fontsize=10)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title(
            f'Confusion Matrix with confidence {confidence} and threshold1:{Init.config["thresh1"]} threshold2:{Init.config["thresh2"]} a:{Init.config["a"]:.2f} b:{Init.config["b"]:.2f} c:{Init.config["c"]:.2f}')
        thresh = self.matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(self.matrix[y, x])
                plt.text(x, y, info, verticalalignment='center', horizontalalignment='center',
                         color="white" if info > thresh else "black",
                         fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_directory, f'confusion_matrix_confidence_{confidence}.png'))  # Save the plot as an image file
        plt.close()  # Close the plot to free up memory

    def save(self, experiment_directory, confidence):
        # Save the confusion matrix as a CSV file
        matrix_df = pd.DataFrame(self.matrix, index=self.labels, columns=self.labels)
        matrix_df.to_csv(os.path.join(experiment_directory, f'confusion_matrix_confidence_{confidence}.csv'),
                         index=True)


class Init:
    # Create configuration dictionary
    config = dict(
        experiment_name="test_of_test_set_trained_224_lr_bs32_with_rejection",
        MedMNIST_dataset_name="DermaMNIST",
        image_size=224,
        model_weight_path='Train_runs/Train_2025-02-22_14-08-32/medmamba_t_Net.pth',
        transform=dict(
            resize=(224, 224),
            normalize=dict(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ),
        batch_size=32,
        test_net="medmamba_t",  # Choose the appropriate one
        # input "single_confidence_value" for testing a single confidence value or "scan_confidence_values" for multiple
        test_type="scan_confidence_values",
        confidence=100,
        confidence_values=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        thresh1=False,
        thresh2=True,
        a=-5.9,
        b=942,
        c=36.4
    )


def setup():
    # Dataset name to class mapping
    dataset_class_mapping = {
        "DermaMNIST": DermaMNIST,
        # Add other dataset mappings as needed
    }

    # Use the mapping to get a reference to the dataset class
    dataset_class = dataset_class_mapping[Init.config["MedMNIST_dataset_name"]]

    experiment_name = Init.config["experiment_name"]
    # Create experiment directory based on current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_directory = f"Test_runs/{timestamp}_{experiment_name}"  # Directory path
    os.makedirs(experiment_directory, exist_ok=True)  # Create the directory

    # Save the configuration file in the experiment directory
    config_file_path = os.path.join(experiment_directory, 'config.json')
    with open(config_file_path, 'w') as json_file:
        json.dump(Init.config, json_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = {
        "test": transforms.Compose([
            transforms.Resize(Init.config["transform"]["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=Init.config["transform"]["normalize"]["mean"],
                                 std=Init.config["transform"]["normalize"]["std"])
        ])
    }

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
    #                                         transform=data_transform)

    test_dataset = dataset_class(split="test", download=True, size=Init.config["image_size"],
                                 transform=data_transform["test"])

    batch_size = Init.config["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=2)
    labels = [v for k, v in test_dataset.info['label'].items()]
    nc = len(labels)
    return nc, labels, device, test_loader, experiment_directory, test_dataset


def main(nc, labels, device, test_loader, experiment_directory):
    if Init.config["test_net"] == "medmamba_t":
        net = VSSM(Init.config["confidence"], Init.config["thresh1"], Init.config["thresh2"], Init.config["a"],
                   Init.config["b"], Init.config["c"], depths=[2, 2, 4, 2], dims=[96, 192, 384, 768],
                   num_classes=nc).to(device)
    elif Init.config["test_net"] == "medmamba_s":
        net = VSSM(Init.config["confidence"], Init.config["thresh1"], Init.config["thresh2"], Init.config["a"],
                   Init.config["b"], Init.config["c"], depths=[2, 2, 8, 2], dims=[96, 192, 384, 768],
                   num_classes=Init.nc).to(device)
    elif Init.config["test_net"] == "medmamba_b":
        net = VSSM(Init.config["confidence"], Init.config["thresh1"], Init.config["thresh2"], Init.config["a"],
                   Init.config["b"], Init.config["c"], depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024],
                   num_classes=nc).to(device)

    # load pretrain weights
    model_weight_path = Init.config["model_weight_path"]
    assert os.path.exists(model_weight_path), f"cannot find {model_weight_path} file"
    weights = torch.load(model_weight_path, map_location=device)
    net.load_state_dict(weights)
    net.to(device)

    confusion = ConfusionMatrix(num_classes=nc, labels=labels)
    rejection_count = 0
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(test_loader):
            val_images, val_labels = val_data
            softmax_probs, preds, should_reject = net(val_images.to(device))
            # Convert torch tensor to numpy
            preds = preds.cpu().numpy()
            should_reject = should_reject.cpu().numpy()
            val_labels = val_labels.cpu().numpy()
            softmax_probs = softmax_probs.cpu().numpy()
            rejection_count = rejection_count + np.sum(should_reject)
            # Use unaltered softmax_probs for AUC calculation
            scores = softmax_probs
            # Handle rejection
            preds[should_reject] = confusion.num_classes - 1  # Assign rejection class index (last index)
            # Debug information
            print(f"Predictions after applying rejection logic: {preds}")
            print(confusion.matrix)  # State of the matrix before update
            confusion.update(preds, val_labels, scores)
            #scores = torch.softmax(outputs, dim=1)  # Use scores for AUC
            #preds = torch.argmax(scores, dim=1)
            #confusion.update(preds.to("cpu").numpy(),
            #                 val_labels.to("cpu").numpy(),
            #                 scores.to("cpu").numpy())

    print(f"Rejections in this batch: {rejection_count}")

    confusion.save(experiment_directory, Init.config["confidence"])
    confusion.plot(experiment_directory, Init.config["confidence"])
    confusion.summary(experiment_directory, Init.config["confidence"])


def run_tests_with_confidence(confidence_values, nc, labels, device, test_loader, test_dataset):
    valid_accuracy_count = 0
    results = {}

    for confidence in confidence_values:
        # Modify the config dictionary for the current confidence
        Init.config["confidence"] = confidence

        # Initialize the model with the current confidence
        if Init.config["test_net"] == "medmamba_t":
            net = VSSM(Init.config["confidence"], Init.config["thresh1"], Init.config["thresh2"],
                       Init.config["a"], Init.config["b"], Init.config["c"], depths=[2, 2, 4, 2],
                       dims=[96, 192, 384, 768], num_classes=nc).to(device)
        elif Init.config["test_net"] == "medmamba_s":
            net = VSSM(Init.config["confidence"], Init.config["thresh1"], Init.config["thresh2"],
                       Init.config["a"], Init.config["b"], Init.config["c"], depths=[2, 2, 8, 2],
                       dims=[96, 192, 384, 768], num_classes=nc).to(device)
        elif Init.config["test_net"] == "medmamba_b":
            net = VSSM(Init.config["confidence"], Init.config["thresh1"], Init.config["thresh2"],
                       Init.config["a"], Init.config["b"], Init.config["c"], depths=[2, 2, 12, 2],
                       dims=[128, 256, 512, 1024], num_classes=nc).to(device)

        # Load pretrain weights
        model_weight_path = Init.config["model_weight_path"]
        assert os.path.exists(model_weight_path), f"cannot find {model_weight_path} file"
        weights = torch.load(model_weight_path, map_location=device)
        net.load_state_dict(weights)
        net.to(device)

        # Initialize Confusion Matrix
        confusion = ConfusionMatrix(num_classes=nc, labels=labels)

        # Overall variables to track
        total_samples = len(test_dataset)  # This is the total number of samples in your dataset
        rejection_count = 0
        total_predictions = 0
        overall_accuracy_non_rejected = 0

        net.eval()
        with torch.no_grad():
            for val_data in tqdm(test_loader):
                val_images, val_labels = val_data
                softmax_probs, preds, should_reject = net(val_images.to(device))
                preds = preds.cpu().numpy()
                should_reject = should_reject.cpu().numpy()
                val_labels = val_labels.cpu().numpy()
                softmax_probs = softmax_probs.cpu().numpy()

                # Handle rejection
                preds[should_reject] = confusion.num_classes - 1  # Assign rejection class index

                # Update confusion matrix
                confusion.update(preds, val_labels, softmax_probs)

                # Track rejection statistics
                rejection_count += np.sum(should_reject)

        # Calculate percentage of rejections
        rejection_percentage = (rejection_count / total_samples) * 100

        # Calculate overall accuracy excluding rejections
        valid_preds = preds[preds != confusion.num_classes - 1]  # Exclude rejected predictions
        valid_labels = val_labels[preds != confusion.num_classes - 1]
        valid_labels = valid_labels.squeeze()  # Flatten the labels to remove extra dimension
        # Calculate overall accuracy excluding rejections
        if len(valid_labels) > 0:
            valid_accuracy_count = np.sum(valid_preds == valid_labels)
            overall_accuracy_non_rejected = confusion.summary(experiment_directory, confidence) #valid_accuracy_count / len(valid_labels)
        else:
            overall_accuracy_non_rejected = 0

        print(f'Valid Accuracy Count: {valid_accuracy_count}, Total Valid: {len(valid_labels)}')
        print(f'Valid Predictions Shape: {valid_preds.shape}')
        print(f'Valid Labels Shape: {valid_labels.shape}')
        print(f'Valid Predictions: {valid_preds}')
        print(f'Valid Labels: {valid_labels}')

        # Store results for plotting
        results[confidence] = {
            "rejection_percentage": rejection_percentage,
            "overall_accuracy": overall_accuracy_non_rejected
        }
        confusion.save(experiment_directory, confidence)
        confusion.plot(experiment_directory, confidence)
        confusion.summary(experiment_directory, confidence)
    return results


if __name__ == '__main__':
    nc, labels, device, test_loader, experiment_directory, test_dataset = setup()
    if Init.config["test_type"] == "single_confidence_value":
        main(nc, labels, device, test_loader, experiment_directory)
    if Init.config["test_type"] == "scan_confidence_values":
        results = run_tests_with_confidence(Init.config["confidence_values"], nc, labels, device, test_loader,
                                            test_dataset)

        # Prepare data for plotting directly from results
        rejection_percentages = [results[confidence]['rejection_percentage'] for confidence in
                                 Init.config["confidence_values"]]
        overall_accuracies = [results[confidence]['overall_accuracy'] for confidence in
                              Init.config["confidence_values"]]

        # Calculate and display AUC
        auc = auc(rejection_percentages, overall_accuracies)
        print(f'AUC of the rejection vs accuracy curve: {auc:.3f}')
        # Plot rejection percentage vs. overall accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(rejection_percentages, overall_accuracies, marker='o', linestyle='-')
        plt.title(f'Rejection Percentage vs. Overall Accuracy, AUC:{auc:.3f}')
        plt.xlabel('Rejection Percentage (%)')
        plt.ylabel('Overall Accuracy')
        plt.grid(True)

        # Annotate each data point with its confidence value
        for i, confidence in enumerate(Init.config["confidence_values"]):
            plt.text(rejection_percentages[i], overall_accuracies[i], f'{confidence}%', fontsize=9, ha='right')

        # Save the plot
        plt.savefig(os.path.join(experiment_directory, 'rejection_accuracy_plot.png'))  # Save the plot
        plt.close()  # Close the plot to free memory
