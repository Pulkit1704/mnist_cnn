import matplotlib.pyplot as plt 

from cnn_model import Model 
from data_utils import load_mnist 
import torch 
import torchvision 


transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

_, testloader = load_mnist(root= "./data", batch_size = 1, transformation= transformation) 

data = next(iter(testloader)) 

inputs, labels = data 

sample_image = inputs[0] 
model = Model(num_channels=10, kernel_size=3, pool_size=2)  

model.load_state_dict(torch.load("./trained_model/mnist_trained.pth"))


def filter_show(layer_output, axes_arr): 

    i, j = 0, 0 
    for filter in layer_output: 
        if i >= axes_arr.shape[0]: 
            j += 1 
            i = 0 
        
        axes_arr[i, j].imshow(filter.detach().numpy()) 

        i+=1  

plt.imshow(inputs[0].numpy()[0]) 

l1 = model.act1(model.conv1(sample_image))
fig1 = plt.figure(figsize=(20, 20)) 

axes1 = fig1.subplots(2, 5) 

filter_show(l1, axes1) 

l2 = model.act2( model.conv2(l1))

fig2 = plt.figure(figsize=(20, 20)) 

axes2 = fig2.subplots(2, 5) 

filter_show(l2, axes2)
 
plt.show() 