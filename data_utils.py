import torch
import torchvision 
from torch.utils.data import DataLoader

def load_mnist(root, batch_size, transformation, shuffle = True): 

    train_data = torchvision.datasets.MNIST(root, train = True, download= True, transform= transformation) 

    train_loader = DataLoader(train_data, batch_size, shuffle) 

    test_data = torchvision.datasets.MNIST(root, train = False, download= False, transform= transformation) 

    test_loader = DataLoader(test_data, batch_size, shuffle) 

    return train_loader, test_loader 


