import torch
from torchvision import models
from PIL import Image
import numpy as np
from cli_predict_args import get_args
import os
def main():
    
    parser = get_args()
    
    cli_args = parser.parse_args()
    # check for model file
    if not os.path.isdir(cli_args.checkpoint_file):
        print(f'Saved Model file {cli_args.checkpoint_file} was not found.')
        exit(1)

    # check for test image
    if not os.path.isdir(cli_args.path_to_image):
        print(f'Test Image {cli_args.path_to_image} does not exist. ')
        exit(1)

    filename = cli_args.checkpoint_file
    inputfilename= cli_args.path_to_image #'flowers/test/10/image_07090.jpg'
    
    model_from_file = rebuildModel(filename)
    print(model_from_file)

    topk = cli_args.top_k
    
    top_probs, classes= predict(inputfilename,model_from_file,topk)
    print(top_probs)
    print(classes)

def rebuildModel(filepath):
    
    # Load model metadata
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    model = getattr(models, checkpoint['name'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    # TODO: Implement the code to predict the class from an image file
    # Move model into evaluation mode and to CPU
    model.eval()
    model.cpu()
   
    # Open image
    image = Image.open(image_path)
    
    # Process image
    image = process_image(image) 
    
    # Change numpy array type to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    
    # Format tensor for input into model
    # (add batch of size 1 to image)
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    #print(top_probs)
    #print(top_labs)
    
    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Find the shorter side and resize it to 256 keeping aspect ratio
    # if the width > height
    if image.width > image.height:        
        # Constrain the height to be 256
        image.thumbnail((10000000, 256))
    else:
        # Constrain the width to be 256
        image.thumbnail((256, 10000000))
    
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose(2, 0, 1)
    
    return image

    
if __name__ == '__main__':
    main()