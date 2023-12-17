from io import BytesIO 
import base64 
import torchvision.transforms as transforms 
from model.cnn_model import Model
import torch 


from PIL import Image 


def transform_image(image): 

    transformation = transforms.Compose([
        transforms.Resize(28), 
        transforms.CenterCrop([28, 28]),
        transforms.ToTensor()
    ])
    
    data = image.split(b',')[-1]
    image_data = base64.decodebytes(data) 
    pil_image = Image.open(BytesIO(image_data))
    processed_image = transformation(pil_image) 

    return processed_image


def make_predictions(image_tensor ):

    model = Model(10, 3, 2)

    model.load_state_dict(torch.load("model/trained_model/mnist_trained.pth"))

    predictions_tensor = model.predict(torch.reshape(image_tensor[3], [1, 1, 28, 28]))

    _, predicted_class = torch.max(predictions_tensor, dim = 1) 

    return predicted_class 