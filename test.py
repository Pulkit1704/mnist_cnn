from cnn_model import Model 
import torch 
from data_utils import load_mnist 
import torchvision 

transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

_, testloader = load_mnist(root = "./data", batch_size= 1000, transformation=transformation) 

model = Model(
    num_channels=10,
    kernel_size=3,
    pool_size=2
) 

model.load_state_dict(torch.load("./trained_model/mnist_trained.pth"))

classified = 0 
for data in testloader: 

    inputs, labels = data 

    outputs = model.predict(inputs) 

    _, predictions = torch.max(outputs, dim = 1) 

    classified += sum(map(int, torch.eq(labels, predictions))) 


accuracy = classified/ 10_000 

print(accuracy) 
