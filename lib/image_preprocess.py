import io 
import torchvision.transforms as transforms 

from PIL import Image 


def transform_image(image_bytes): 

    tranformation = transforms.Compose([
        transforms.Resize(28), 
        transforms.CenterCrop(28),
        transforms.ToTensor()
    ])

    # the tensor right now contains 4 channels for rgba, see which ones are useful to us 
    # the final tensor going for predictions should have a single monochromatic channel. 
    # the last channel has most of the data we need, find what to do with the rest 3. 
    return tranformation(image_bytes) 