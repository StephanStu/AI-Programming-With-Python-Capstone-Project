#-*-coding: utf-8-*-
"""
@author: Nadine Studener
@title: prediction script for Flower Image classification
"""

#----------------------------------------------------------#
#Imports here
from PIL import Image
from collections import OrderedDict
import argparse
import time
import torch
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, models, transforms

def process_image(image):
    image_pil = Image.open(image)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = adjustments(image_pil)
    return image_tensor


def process_image(image):
    image_pil = Image.open(image)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = adjustments(image_pil)
    return image_tensor

def parse_arguments():
    parser = argparse.ArgumentParser(description='classify an image using a neuronal network!')
    parser.add_argument('--image', default="image.jpg", help='image file for classification.')
    parser.add_argument('--checkpoint', default= "checkpoint.pth", help='pre-trained network for classification.')
    parser.add_argument('--top_k', default =5, type=int, help='number of most probable categories.')
    parser.add_argument('--categories', type = str, default="cat_to_name.json", help='category-to-name-mapping as .json.')
    parser.add_argument('--gpu', action='store_true', help='gpu.')
    arguments = parser.parse_args()
    return arguments


arguments = parse_arguments()
if(arguments.top_k is None):
    topk=5
    print("Warning! Top_k must be provided. Setting top_k=5 to proceed!")
else:
    topk = int(arguments.top_k)

if (arguments.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled without GPU available")

with open(arguments.categories, 'r') as f:
    cat_to_name = json.load(f)

#display inputs for preciction
print("Image directory for classification:   " + str(arguments.image))
print("GPU enabled:                          " + str(arguments.gpu))
print("Number of top_k:                      " + str(arguments.top_k))
print("Category to names mapping directory   " + str(arguments.categories))
print("Loading the neuronal network from:    " + str(arguments.checkpoint))

# load the model data from checkpoint
model_data = torch.load(arguments.checkpoint, map_location=lambda storage, loc: storage)

# load a model from https://download.pytorch.org
if (model_data['model'] == 'densenet'):
    model = models.densenet121(pretrained=True)
    input_node = model_data['classifierInputSize']
    output_node = model_data['classifierOutputSize']

if (model_data['model'] == 'vgg16_bn'):
    model = models.vgg16_bn(pretrained =True)
    input_node = model_data['classifierInputSize']
    output_node = model_data['classifierOutputSize']

# exchange classifier and paramterize it with model-data
model.classifier = model_data['classifier']
model.load_state_dict(model_data["state_dict"])
model.class_to_idx = model_data["Index_to_class"]
model.eval()

# pre-process the image, return tensor
image = process_image(arguments.image)

# predict the image
with torch.no_grad():
    #switch from numpy to pytorch
    imageTorch = torch.tensor(image)
    imageTorch = imageTorch.float()
    #remove runtime error
    imageTorch.unsqueeze_(0)
    #run model
    outputs = model(imageTorch)
    probabilities, classes = torch.exp(outputs).topk(topk)
top_probabilities = probabilities[0].tolist()
top_classes = classes[0].add(1).tolist()
print("Most probable class:                  " + str(cat_to_name[str(top_classes[0])]))
print("Result, Top Probabilities:            " + str(top_probabilities))
print("Result, Top Classes:                  " + str(top_classes))
