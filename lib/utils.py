from io import BytesIO 
import base64 
import torchvision.transforms as transforms 
from model.cnn_model import Model
import torch 
import json 
from PIL import Image 


def transform_image(image): 

    '''
        Takes a Base 64 image from the client and transforms the image into a tensor. 
    '''

    transformation = transforms.Compose([
        transforms.Resize(28), 
        transforms.CenterCrop([28, 28]),
        transforms.ToTensor()
    ])

     
    baset64Data = image.split(b',')[-1]
    image_data = base64.decodebytes(baset64Data) 
    pil_image = Image.open(BytesIO(image_data))
    processed_image = transformation(pil_image) 

    return processed_image


def build_model():

    with open("model/trained_model/hyperparameters.json") as file: 
        hyperparameters = json.load(file) 


    model = Model(
                hyperparameters["num_channels"], 
                hyperparameters["kernel_size"],
                hyperparameters["pool_size"]
    )

    model.load_state_dict(torch.load("model/trained_model/mnist_trained.pth"))

    return model


def make_predictions(image_file, prebuild_model): 

    image_tensor = transform_image(image_file) 

    # the pil image sent from the client contains 4 channels: rgba 
    # we take the alpha channel since this has information we need.
    image_tensor = torch.reshape(image_tensor[-1], [1, 1, 28, 28])

    predictions_tensor = prebuild_model.predict(image_tensor) 


    _, predicted_class = torch.max(predictions_tensor, dim=1) 

    return predicted_class 