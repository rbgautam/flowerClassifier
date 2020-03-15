import numpy as np
import json
import time
import os
import random
from classifier import Classifier

import seaborn as sns
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

from cli_train_args import get_args




def main():
    parser = get_args()
    
    cli_args = parser.parse_args()

    # check for data directory
    if not os.path.isdir(cli_args.data_directory):
        print(f'Data directory {cli_args.data_directory} was not found.')
        exit(1)

    # check for save directory
    if not os.path.isdir(cli_args.save_dir):
        print(f'Directory {cli_args.save_dir} does not exist. Creating...')
        os.makedirs(cli_args.save_dir)

    data_dir = cli_args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    save_dir = cli_args.save_dir
    lr = cli_args.learning_rate
    model_name = cli_args.arch
    hidden_units = cli_args.hidden_units
    epochs = cli_args.epochs
    gpu = cli_args.use_gpu

    print("data_dir = ",data_dir,"\n,save_dir=",save_dir,"\n,lr=",lr,"\n,model_name=",model_name,"\n,hidden_units=",hidden_units,"\n,epochs=",epochs,"\n,GPU=",gpu)

    if model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.name = model_name
    
    if model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.name = model_name

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = model_name
        
    trainloader,validloader,train_data = transform_train(train_dir,valid_dir)
    read_labels()
    run_train(model,trainloader,validloader,train_data,lr,hidden_units,epochs,gpu,save_dir)

def transform_train(train_dir,valid_dir):
    # Dataset values
    image_size = 224 # Image size in pixels
    reduction = 255 # Image reduction to smaller edge 
    norm_means = [0.485, 0.456, 0.406] # Normalized means of the images
    norm_std = [0.229, 0.224, 0.225] # Normalized standard deviations of the images
    rotation = 45 # Range of degrees for rotation
    batch_size = 64 # Number of images used in a single pass
    shuffle = True # Randomize image selection for a batch
    
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(image_size),
                                       transforms.RandomRotation(rotation),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(norm_means, norm_std)])
    validate_transforms = transforms.Compose([transforms.Resize(reduction),
                                      transforms.CenterCrop(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_means, norm_std)])
    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=validate_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    return trainloader,validloader,train_data
    
def read_labels():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    print(f"Images are labeled with {len(cat_to_name)} categories.")
    return cat_to_name
    
def run_train(model,trainloader,validloader,train_data,lr,hidden_units,epochs,gpu,save_dir):
    input_size = 25088
    output_size = 102
    hidden_layers = [hidden_units[0], 1024]
    drop_out = 0.2
    model.classifier = Classifier(input_size, output_size, hidden_layers, drop_out)
    if gpu:
        current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        current_device = torch.device("cpu")
    print(current_device)
    # Define the loss function
    criterion = nn.NLLLoss()

    # Define weights optimizer (backpropagation with gradient descent)
    # Only train the classifier parameters, feature parameters are frozen
    # Set the learning rate as lr=0.001
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    # Move the network and data to GPU or CPU
    model.to(current_device)
    print(model)
    drop_out = 0.2
    learning_rate = lr
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs_no = epochs
    
    train_loss, valid_loss, valid_accuracy = trainClassifier( model, epochs_no, criterion, optimizer, trainloader,validloader, current_device)

    print("Final result \n",
          f"Train loss: {train_loss:.3f}.. \n",
          f"Test loss: {valid_loss:.3f}.. \n",
          f"Test accuracy: {valid_accuracy:.3f}")
    filename = saveCheckpoint(model,train_data,save_dir)
    print(filename)
    
def saveCheckpoint(model,train_data,save_dir):
    
    # Mapping of classes to indices
    model.class_to_idx = train_data.class_to_idx
    
    # Create model metadata dictionary
    checkpoint = {
        'name': model.name,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict()
    }
    print(model.classifier)
    # Save to a file
    timestr = time.strftime("%m%d%Y_%H%M%S")
    file_name = save_dir + 'classify_model_' + timestr + '.pth'
    torch.save(checkpoint, file_name)
    return file_name   

#A function used for validation and testing
def testClassifier(model, criterion, testloader, current_device):
    
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
        
    test_loss = 0
    accuracy = 0
        
    # Looping through images, get a batch size of images on each loop
    for inputs, labels in testloader:

        # Move input and label tensors to the default device
        inputs, labels = inputs.to(current_device), labels.to(current_device)

        # Forward pass, then backward pass, then update weights
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)
        test_loss += batch_loss.item()

        # Convert to softmax distribution
        ps = torch.exp(log_ps)
        
        # Compare highest prob predicted class with labels
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        # Calculate accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return test_loss, accuracy
# A function used for training (and tests with different model hyperparameters)
def trainClassifier(model, epochs_no, criterion, optimizer, trainloader, validloader, current_device):
    
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
    
    epochs = epochs_no
    steps = 0
    print_every = 1
    running_loss = 0

    # Looping through epochs, each epoch is a full pass through the network
    for epoch in range(epochs):
        
        # Switch to the train mode
        model.train()

        # Looping through images, get a batch size of images on each loop
        for inputs, labels in trainloader:

            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(current_device), labels.to(current_device)

            # Clear the gradients, so they do not accumulate
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Track the loss and accuracy on the validation set to determine the best hyperparameters
        if steps % print_every == 0:

            # Put in evaluation mode
            model.eval()

            # Turn off gradients for validation, save memory and computations
            with torch.no_grad():

                # Validate model
                test_loss, accuracy = testClassifier(model, criterion, validloader, current_device)
                
            train_loss = running_loss/print_every
            valid_loss = test_loss/len(validloader)
            valid_accuracy = accuracy/len(validloader)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss:.3f}.. "
                  f"Test loss: {valid_loss:.3f}.. "
                  f"Test accuracy: {valid_accuracy:.3f}")

            running_loss = 0
            
            # Switch back to the train mode
            model.train()
                
    # Return last metrics
    return train_loss, valid_loss, valid_accuracy


if __name__ == '__main__':
    main()