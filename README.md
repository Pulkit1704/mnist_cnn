# Convolutional Neural Network trained on the MNIST digit classification dataset. 

## Requried packages: 
* pytorch 
* matplotlib 
* torchvision 
* flask

The environment plan is located in the environment.yml file. 

## Project Description 
This is a pytorch implementation of convolutional neural network on the MNIST digit classification dataset. 

The model is stored in the model directory and the server code is storec in the server.py and lib directory. The html templates for the front end are stored in the templates directory and the css and javascript files are stored in the static directory. 

Follow the below instructions to initialize the flask server. 
 

### Instructions to run the program: 
* Fork and clone the repository. Then change into the cloned repository 
```bash
    cd mnist_cnn
```
* Build the conda environment using the environment.yml file.

```bash 
    conda env create -f environment.yml
``` 
* Run the server.py file using the below command 

```bash 
    python server.py 
```
* Visit localhost:8080 on your local machine to interact with the server. 