# Convolutional Neural Network trained on the MNIST digit classification dataset. 

python version: 3.11.3

## Requried packages: 
* pytorch 
* matplotlib 
* torchvision 

The environment plan is in environment.yml files for conda environment creation. 

## Project Description 
This is a pytorch implementation of convolutional neural network on the MNIST digit classification database. The model architecture is summarised in the below image. 

You can train and test the model using the train.py and test.py respectively by running them from the root of the project folder. The dataset is stored in the data directory.

The model is a simplified convolutional network with convolution layers implementing same convolution on input images. There are 2 convolution layers with a user definable number of convolution filters in the form of num_channels. The convoluted input is flattened and fed into a feed forward neural network with progressively decreasing number of neurons in each layer until it reaches to an output layer with 10 neurons. 

### Instructions to run the program: 
* Fork and clone the repository. 
* Build the conda environment using the environment.yml file. 
* run the test.py file from the created environment. 