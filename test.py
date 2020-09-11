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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_class = 24
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder('gray_testing/', data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8)
    print(test_dataset.class_to_idx)
    confusion_matrix = torch.zeros(nb_class, nb_class)
    testing_model, input_size = initialize_model('resnet18', nb_class, False)
    testing_model.eval()
    testing_model = testing_model.to(device)
    testing_model.load_state_dict(torch.load('model/resnet18_24/model-29.ckpt'))
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = testing_model(inputs)
            _, pred = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print("confusion_matrix", confusion_matrix)
    print("per class accuracy:", confusion_matrix.diag() / confusion_matrix.sum(1))
