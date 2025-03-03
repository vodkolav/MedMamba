import datetime
import os
import sys
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from MedMamba import VSSM as medmamba  # import model
from medmnist import INFO, DermaMNIST

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print(os.getcwd())
    # Create configuration dictionary
    config = dict(experiment_name="Train_lr_change_image_224_bs_32",
                  MedMNIST_dataset_name="DermaMNIST",
                  train_net="medmamba_t",
                  fine_tune=False,
                  image_size= 224,
                  model_weight_path ='Train_runs/Train_lr_change_image_224_2025-03-03_00-47-47/medmamba_t_Net.pth',
                  transform= dict(RandomResizedCrop=(224),
                                  resize=(224, 224),
                                  normalize={"mean": [0.5, 0.5, 0.5],
                                              "std": [0.5, 0.5, 0.5]}),
                  batch_size=32,
                  learning_rate=0.001,
                  epochs=100
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

    labels = [v for k, v in train_dataset.info['label'].items()]
    nc = len(labels)

    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

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

    if config["train_net"] == "medmamba_t":
        net = medmamba(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=nc)
    elif config["train_net"] == "medmamba_s":
        net = medmamba(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=nc)
    elif config["train_net"] == "medmamba_b":
        net = medmamba(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=nc)

        # load pretrain weights
    if config["fine_tune"] is True:
        model_weight_path = config["model_weight_path"]
        assert os.path.exists(model_weight_path), f"cannot find {model_weight_path} file"
        weights = torch.load(model_weight_path, map_location=device)
        net.load_state_dict(weights)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=config["learning_rate"])

    epochs = config["epochs"]
    best_acc = 0.0
    save_path = os.path.join(experiment_directory, f'{config["train_net"]}_Net.pth')
    metrics = {
        "train_loss": [],
        "val_accuracy": [],
    }
    for epoch in range(config["epochs"]):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            # Adjust learning rate based on specific epochs
            if epoch == 50 or epoch == 75:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1  # Reduce learning rate by a factor of 0.1
            images, labels = data
            labels = labels.squeeze() #TODO: this should be done in dataloader
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accuracy = acc / val_num
        metrics["train_loss"].append(running_loss / len(train_loader))
        metrics["val_accuracy"].append(val_accuracy)
        print(
            f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  val_accuracy: {val_accuracy:.3f}')
        #print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #     (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

        # Plot loss and accuracy after each epoch
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(metrics["train_loss"], label='Train Loss', color='tab:red')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(metrics["val_accuracy"], label='Validation Accuracy', color='tab:blue')
        plt.title('Validation Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save the plot as an image file in the experiment directory
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_directory, f'loss_accuracy_epoch.png'))
        plt.close()  # Close plot to free memory

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(experiment_directory, 'training_metrics.csv'), index=False)
    print('Finished Training')


if __name__ == '__main__':
    main()
