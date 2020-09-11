import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import copy
import torch
from model import initialize_model
from train import train_model
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

REBUILD_DATA = True

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The available device is: ", device)
    model_names = ["resnet18","resnet50"]
    num_classes = 24
    feature_extract = False
    num_epochs = 40

    # Import training set and validation set
    input_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_dataset = ImageFolder('train/', data_transforms['train'])
    val_dataset = ImageFolder('val/', data_transforms['val'])
    print(train_dataset.class_to_idx)
    print(val_dataset.class_to_idx)
    dataloaders_dict = {'train': DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8),
                        'val': DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)}

    # Train and evaluate
    for model_name in model_names:
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        model_ft = model_ft.to(device)
        params_to_update = model_ft.parameters()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        model_ft, hist = train_model(model_name, model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

        #
        # scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
        # scratch_model = scratch_model.to(device)
        # scratch_optimizer = optim.Adam(scratch_model.parameters(), lr=0.001)
        # scratch_criterion = nn.CrossEntropyLoss()
        # _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs)

        # Plot the training curves of validation accuracy vs. number
        #  of training epochs for the transfer learning method and
        #  the model trained from scratch
        ohist = []
        shist = []

        ohist = [h.cpu().numpy() for h in hist]
        # shist = [h.cpu().numpy() for h in scratch_hist]

        plt.title("Validation Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Validation Accuracy")
        plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
        # plt.plot(range(1,num_epochs+1),shist,label="Scratch")
        plt.ylim((0, 1.))
        plt.xticks(np.arange(1, num_epochs + 1, 1.0))
        plt.legend()
        plt.show()
