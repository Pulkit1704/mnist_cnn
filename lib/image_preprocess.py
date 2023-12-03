from io import BytesIO 
import base64 
import torchvision.transforms as transforms 

from PIL import Image 


def transform_image(image): 

    transformation = transforms.Compose([
        transforms.Resize(28), 
        transforms.CenterCrop([28, 28]),
        transforms.ToTensor()
    ])

    # the tensor right now contains 4 channels for rgba, see which ones are useful to us 
    # the final tensor going for predictions should have a single monochromatic channel. 
    # the last channel has most of the data we need, find what to do with the rest 3. 
    data = image.split(b',')[-1]
    image_data = base64.decodebytes(data) 
    pil_image = Image.open(BytesIO(image_data))
    processed_image = transformation(pil_image) 

    return processed_image