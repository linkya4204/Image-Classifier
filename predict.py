import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

import torchvision.models as models
import json
from PIL import Image


parser = argparse.ArgumentParser(description='Predict the category of an image')
    
# Create command line arguments using add_argument() from ArguementParser method
parser.add_argument('--gpu', type = bool, default=True, 
                    help='Specify if you want to run on GPU')
parser.add_argument('--image_path', type = str ,default="flowers/test/1/image_06743.jpg", 
                    help='Specify the image location')
parser.add_argument('--checkpoint', type = str ,default="FloralImageModel.pt", 
                    help='Specify the checkpoint file name')
parser.add_argument('--category_file', default= 'cat_to_name.json',type = str, help='File with mapping category names')


args = parser.parse_args()
im_filepath = args.image_path
checkpoint = args.checkpoint
cat_filepath = args.category_file


# LOAD THE CHECKPOINT
def load_checkpoint(checkpoint="FloralImageModel.pt"):
    checkpoint = torch.load(checkpoint)
    if checkpoint['model'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.densenet161(pretrained=True)
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = checkpoint['epochs']
    loss = checkpoint['loss']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# PROCESS IMAGE
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    img = transform(image)
    return img

# PREDICT IMAGE
def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
        image = image.view(1,3,224,224)
        y_pred = model(image)
        ps = torch.exp(y_pred)
        top_p, top_class = ps.topk(topk, dim=1)
        
    model.train()
    
    #Convert to list
    top_p, top_class = top_p.tolist(), top_class.tolist()
    top_p = sum(top_p,[])
    top_class = sum(top_class,[])
    
    return top_p, top_class


#LABEL MAPPING
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# LOADING CHECKPOINT AND SHOW THE RESULTS
print("Start loading checkpoint")
model = load_checkpoint()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("The model is loaded successfully.")

# Mapping class_to_idx with category names from json file
class_to_idx  = model.class_to_idx
category_dict = {}
for label, value in class_to_idx.items():
    category_dict[value] = cat_to_name[label]

print("Start predicting...")    
prob,classes = predict(im_filepath, model)
print("The predicted flower category: ", category_dict[classes[0]])



