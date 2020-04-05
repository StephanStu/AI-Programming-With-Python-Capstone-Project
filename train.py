#-*-coding: utf-8-*-
"""
@author: Nadine Studener
@title: training script for Flower Image classification
"""

#----------------------------------------------------------#
#Imports here
import time
import json
import torch
import argparse
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torchvision import datasets, models, transforms

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network with options!')
    parser.add_argument('--data_dir', type = str, default="flowers", help='data directory.')
    parser.add_argument('--categories', type = str, default="cat_to_name.json", help='category-to-name-mapping as .json.')
    parser.add_argument('--arch', type = str, default = 'vgg16_bn', help='supprted architectures: vgg16_bn or densenet.')
    parser.add_argument('--learning_rate', type = float, default = 0.002, help='learning rate.')
    parser.add_argument('--hidden_units', type = int, default = 512, help='number of hidden units.')
    parser.add_argument('--epochs', type = int, default = 1, help='epochs.')
    parser.add_argument('--gpu', action='store_true', help='gpu.')
    arguments = parser.parse_args()
    return arguments

def validate_on_testdata(testloader, this_model):
    correct = 0
    total = 0
    this_model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = this_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


arguments = parse_arguments()

if(arguments.epochs<=0):
    print("Warning: Epochs must be > 0. Enforcing epochs = 1 to proceed!")

if (arguments.arch is None):
    arch_type = 'vgg16_bn'

if (arguments.arch != 'vgg16_bn') and (arguments.arch != 'densenet'):
    print("Warning: Network must be vgg16_bn or densenet. Enforcing arch=vgg16_bn to proceed!")
    arguments.arch = 'vgg16_bn'

if(arguments.hidden_units <= 0):
    print("Warning: hidden units must be > 0. Enforcing hidden units= 512 to proceed!")

if(arguments.learning_rate <= 0):
    print("Warning: Learning rate must be > 0. Enforcing learning_rate=0.05 to proceed!")
    arguments.learning_rate = 0.05

with open(arguments.categories, 'r') as f:
    cat_to_name = json.load(f)

# display inputs for training
print("Training a classifier based on model: " + arguments.arch)
print("Number of hidden units in classifier: " + str(arguments.hidden_units))
print("GPU enabled:                          " + str(arguments.gpu))
print("Reading data from:                    " + arguments.data_dir)
print("Number of epochs:                     " + str(arguments.epochs))
print("Learning rate:                        " + str(arguments.learning_rate))
# define data source folders, asume sub-structure
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# define the transforms for the training, validation, and testing sets
normalize= transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
print("Fetching training data from:          " + train_dir)
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])
print("Fetching testing data from:           " + test_dir)
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])
print("Fetching validation data from:        " + valid_dir)
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])

# load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
# using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle= True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=75)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=75)
# load a model from https://download.pytorch.org
if (arguments.arch == 'vgg16_bn'):
    model = models.vgg16_bn(pretrained =True)
    input_node=25088
    output_node = len(cat_to_name)

if (arguments.arch == 'densenet'):
    model = models.densenet121(pretrained=True)
    input_node=1024
    output_node = len(cat_to_name)

# turn off gradients to not do backpropagation through them
for param in model.parameters():
    param.requires_grad = False

# define classifier with dropout probability
dropout = 0.1
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_node, arguments.hidden_units)),
                          ('out1',nn.Dropout(dropout)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(arguments.hidden_units, arguments.hidden_units)),
                          ('out2',nn.Dropout(dropout)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(arguments.hidden_units, output_node)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
# set weights and bias random weights
classifier.fc1.weight.data.normal_(std=0.01)
classifier.fc2.weight.data.normal_(std=0.01)
classifier.fc3.weight.data.normal_(std=0.01)
# add random bias
classifier.fc1.bias.data.fill_(0)
classifier.fc2.bias.data.fill_(0)
classifier.fc3.bias.data.fill_(0)
# substitute classifier of pretrained model with my classifier
model.classifier = classifier
# define criterion and optimizer
criterion = nn.NLLLoss()
# train the classifier parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
# choose device
if arguments.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")
# initiate device in model
model.to(device)
# print setup
print(classifier, optimizer, criterion, device)

#train the model

print_every = 5
steps = 0
running_loss = 0
for epoch in range(arguments.epochs):
    model.train()
    for images, labels in trainloader:
        #add 1 step with each iteration
        steps += 1
        #put images, labels to whatever device is available
        images, labels = images.to(device), labels.to(device)
        #perform the following steps to train the mymodel
        #zero out gradients
        optimizer.zero_grad()
        #get log probabilities from mymodel
        logps = model(images)
        #get loss
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        #increment running loss
        running_loss += loss.item()
        #test accuracy, loss on test data set to validate mymodel with the following steps
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            #turn model in evaluation mode (dropout is turned off)
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    #keep track of test loss
                    test_loss += criterion(logps, labels).item()
                    #get accuracy and therefore get probabilites with exponential function due to softmax
                    ps = torch.exp(logps)
                    #get top probabilites and classes and check for equality
                    equality = (labels.data == ps.max(dim=1)[1])
                    #update accuracy with equality
                    accuracy+= equality.type(torch.FloatTensor).mean()
            #print it all out and get average values for training loss, test loss, accuracy
            print("Epoch:  {}/{}.. ".format(epoch+1, arguments.epochs),
                  "Training loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Test accuracy: {:.3f}".format(accuracy/len(validloader)))
            #put running loss back to 0 at the end and model back to training mode
            running_loss = 0
            model.train()
# validtae model after training
validate_on_testdata(testloader, model)
# save to checkpoint file
indexToClassDict = {v:k for k, v in trainloader.dataset.class_to_idx.items()}
model.cpu()
print("My Model: ", model.classifier)
print("State dictionnairy keys: ",  model.classifier.state_dict().keys())
checkpoint = {"model": arguments.arch,
             "pretrained": True,
             "classifierInputSize": input_node,
             "classifierOutputSize": output_node,
             "features": model.features,
             "classifier": model.classifier,
             "optimizier": optimizer.state_dict(),
             "state_dict": model.state_dict(),
             "Index_to_class": indexToClassDict
             }
torch.save(checkpoint, "checkpoint.pth")
print("Model saved to checkpoint.pth.")
