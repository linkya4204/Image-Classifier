import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

from workspace_utils import active_session
import torchvision.models as models
import json
import time


parser = argparse.ArgumentParser(description='Help user to process image classifier')
    
# Create command line arguments using add_argument() from ArguementParser method
parser.add_argument('--gpu', type = bool, default=True, 
                    help='Specify if you want to run on GPU')
parser.add_argument('--dir', metavar='N', type = str, nargs='?',default="flowers/", 
                    help='Specify the image folder where you want to train your model')
parser.add_argument('--arch', default= 'alexnet', type = str, help='You can choose between alexnet or densenet')
parser.add_argument('--learning_rate', default= 0.001,type = float, help='Learning rate of the model')
parser.add_argument('--hidden_layer', default= 1024, type = int, action="store",help='Hidden layer')
parser.add_argument('--epochs', default= 5,type = int,help='Epochs Size')

args = parser.parse_args()

#Get the user input information
gpu = args.gpu
data_dir = args.dir
chosen_model = args.arch
learning_rate = args.learning_rate
hidden_layer = args.hidden_layer
epochs_size = args.epochs

#Input the code from the Image Classifier Project
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

validation_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
test_data = datasets.ImageFolder(valid_dir,transform=test_transforms)
validation_data = datasets.ImageFolder(test_dir,transform=validation_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data,batch_size=32)
validation_loader = DataLoader(validation_data,batch_size=32)


    
# Build the Model
output_size = 102

if chosen_model == 'alexnet':
    model = models.alexnet(pretrained=True)
    input_size = 9216
else:
    model = models.densenet161(pretrained=True)
    input_size = 2208
    
if gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for param in model.parameters():
    param.requires_grad = False
    
# Create a classifier:
model.classifier = nn.Sequential(nn.Linear(input_size,hidden_layer),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(hidden_layer,output_size),
                          nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(),lr=learning_rate)
model.to(device)

print("Start the training")
start_time = time.time()
with active_session():
    epochs = epochs_size
    steps =0
    #to have data to plot
    train_losses, valid_losses = [], []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            #Apply the model
            output = model(inputs)
            loss = criterion(output, labels)
            
            #Update the weight
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        
        #Initialize test loss and accuracy
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            model.eval()
                
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                        
                #Apply the model
                y_val = model(inputs)
                batch_loss = criterion(y_val, labels)

                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(y_val)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        train_losses.append(running_loss/len(train_loader))
        valid_losses.append(valid_loss/len(validation_loader))
        
        print(f"Epoch {epoch+1}/{epochs}.. "
        f"Train loss: {running_loss/len(train_loader):.3f}.. "
        f"Validation loss: {valid_loss/len(validation_loader):.3f}.. "
        f"Validation accuracy: {accuracy/len(validation_loader):.3f}")
        #Back to training mode.                    
        model.train()
        

    print(f"\n Duration for training: {time.time() - start_time:0f} seconds.")

# TESTING YOUR NETWORK
model.eval()

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        
        outcome = model(X_test)
        # Calculate accuracy
        ps = torch.exp(outcome)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y_test.view(*top_class.shape)
        correct += torch.mean(equals.type(torch.FloatTensor)).item()
        
print(f"Test accuracy for the model: {correct/len(test_loader)}")

# SAVE THE CHECKPOINT
model.class_to_idx = train_data.class_to_idx
state = {
    'model':chosen_model,
    'input_size':input_size,
    'hidden_layer':hidden_layer,
    'output_size': output_size,
    'class_to_idx': model.class_to_idx,
    'classifier': model.classifier,
    'epochs': epochs,
    'state_dict':model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': loss,
}

if chosen_model == 'alexnet':
    torch.save(state, 'FloralImageModel.pt')
else:
    torch.save(state, 'FloralImage_DensetNetModel.pt')

print(f"The checkpoint is saved.")





