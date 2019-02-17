import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from load_data import process_image
from load_data import imshow

# Set up model
def setup_model(structure = 'vgg16', dropout = 0.5, lr=0.001, power = 'GPU', hidden_layer = 512):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[0].in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(p=dropout)),
        ('fc1', nn.Linear(num_features, hidden_layer)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer, 100)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(100,102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    if torch.cuda.is_available() and power == 'GPU':
        model.cuda()
    return model, criterion, optimizer

# Train model using your setting
def train_model(model, criterion, optimizer, train_dataloaders, vaild_dataloaders, power = 'GPU', epochs=1):
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in train_dataloaders:
            if torch.cuda.is_available() and power=='GPU':
                images,labels = images.to('cuda'), labels.to('cuda')
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            model.eval()
            accuracy = 0
            vaild_loss = 0
            with torch.no_grad():
                for images, labels in vaild_dataloaders:
                    if torch.cuda.is_available() and power=='GPU':
                        images,labels = images.to('cuda'), labels.to('cuda')
                    log_ps = model(images)
                    vaild_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            running_loss = running_loss / len(train_dataloaders)
            valid_loss = vaild_loss / len(vaild_dataloaders)
            accuracy = accuracy / len(vaild_dataloaders)
            print("epoch {0}/{1} Training loss: {2} ".format(e+1, epochs, running_loss),
                  "Vaildation loss: {}".format(valid_loss),
                  "Accurancy:{}".format(accuracy))
    print('Finished!')

# Test your model
def test_model(model, test_dataloaders, criterion, power = 'GPU'):
    size = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloaders:
            if torch.cuda.is_available() and power=='GPU':
                images,labels = images.to('cuda'), labels.to('cuda')
            log_ps = model(images)
            vaild_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            size += labels.size(0)
            correct += equals.sum().item()
    test_accuracy = correct / size
    print("Accurancy:{:.4f}".format(test_accuracy))

# Save model
def save_model(class_to_idx, path, model, structure, optimizer):
    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
        'optimizer': optimizer,
        'arch': structure,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint,path)
    print('Trained model has been saved!')

# Load model
def load_model(path = 'checkpoint.pth'):
    checkpoint = torch.load(path)
    classifier = checkpoint['classifier']
    structure = checkpoint['arch']
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(image_path, model, topk=5, power = 'GPU', category_names = 'cat_to_name.json'):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    model.eval()
    model.to('cuda')
    img = process_image(image_path)
    if torch.cuda.is_available() and power=='GPU':
        img = img.to('cuda')
    img = img.unsqueeze_(0)
    img = img.float()
    with torch.no_grad():
        output = model.forward(img)
    probability = torch.exp(output)
    prob, label = probability.topk(topk)
    folder_index = []
    for i in np.array(label[0]):
        for folder, num in model.class_to_idx.items():
            if num == i:
                folder_index += [folder]
    flower_name = [cat_to_name[i] for i in folder_index]
    return prob, flower_name






    

