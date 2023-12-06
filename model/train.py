import torch 
import torch.nn as nn 
import torchvision 
import torch.optim as optim 
from cnn_model import Model 
import os 
import matplotlib.pyplot as plt
import json 
import logging 
from data_utils import load_mnist 


# model_save, plot_save and hyper_parameter_save are helper function to save: 
# model state
# training plots 
# hypter parameters used for training. 
def model_save(save_dir, model_state): 

    filename = "mnist_trained.pth"
    path = os.path.join(save_dir, filename) 

    torch.save(model_state, path) 

    logging.info("model_saved...") 

def plot_save(save_dir, loss_vec, accuracy_vec): 

    plt.plot(range(len(loss_vec)), loss_vec, color = "blue", marker = 'o', label = "training error") 
    plt.grid() 
    plt.xlabel("no of epochs") 
    plt.ylabel("training error") 
    plt.title("convergence of training error") 
    plt.savefig(os.path.join(save_dir, "model training loss.png"))  
    plt.close() 

    plt.plot(range(len(accuracy_vec)), accuracy_vec, color = "red", marker = 'o', label = "test set accuracy") 
    plt.grid() 
    plt.xlabel("no of epochs")
    plt.ylabel("test accuracy") 
    plt.title("model accuracy convergence") 
    plt.savefig(os.path.join(save_dir, "model_accuracy_score.png")) 
    plt.close() 

    logging.info("loss and accuracy plots saved.")

def hyper_parameter_save(save_dir, hyper_params): 

    hyper_param_file = os.path.join(save_dir, "hyperparameters.json") 


    with open(hyper_param_file, 'w') as file: 
        json.dump(hyper_params, file, indent = 4) 
    
    logging.info("model hyperparamters saved") 

# wrapper function to save all the important information from a single training session
def session_save(model, save_dir, loss_vec, accuracy_vec, hyper_parameters): 

    logging.info("making the save directory") 
    try: 
        os.mkdir(save_dir)
    except FileExistsError: 
        logging.warning(f"{save_dir} already exists, saving the files. ")

    model_save(save_dir, model) 

    plot_save(save_dir,loss_vec, accuracy_vec) 

    hyper_parameter_save(save_dir, hyper_parameters) 


####################################################################################################
if __name__ == '__main__': 

    logging.basicConfig(level = logging.INFO)

    logging.info("loading the data and building model...")

    transformation = torchvision.transforms.ToTensor() 

    train_loader, test_loader = load_mnist(root = "./model/data", batch_size=64, transformation= transformation) 

    hyper_parameters = {
        "num_channels": 10,
        "kernel_size": 3,
        "pool_size": 2,
        "learning_rate": 0.01, 
        "momentum": 0.8, 
        "max_epochs": 100, 
        "accuracy_increment_count": 5,
        "max_accuracy": 0.0
    }


    model= Model(num_channels= hyper_parameters["num_channels"],
                kernel_size=hyper_parameters['kernel_size'],
                pool_size=hyper_parameters['pool_size'])


    loss = nn.CrossEntropyLoss() 

    optimizer = optim.SGD(model.parameters(), lr = hyper_parameters["learning_rate"],
                          momentum = hyper_parameters["momentum"]) 


    n_epochs = 0 
    per_epoch_loss = [] 
    per_epoch_accuracy = [0.0] 
    accuracy_count = 0 

    max_accuracy_state = {} 

    logging.info("starting training...") 

    while accuracy_count < hyper_parameters["accuracy_increment_count"] and n_epochs < hyper_parameters["max_epochs"]:

        logging.info(f"starting epoch {n_epochs}")

        #training 
        total_training_loss = model.model_train(loss,
                                          optimizer,
                                          train_loader) 
        
        logging.info(f"epoch {n_epochs} training done.. model loss.. {total_training_loss}")

        per_epoch_loss.append(total_training_loss) 

        # testing 
        accuracy = model.model_test(test_loader) 

        logging.info(f"model testing done.. model accuracy.. {accuracy}")

        if accuracy <= hyper_parameters["max_accuracy"]:
            accuracy_count += 1
        else: 
            accuracy_count = 0 
            hyper_parameters["max_accuracy"] = accuracy 

            max_accuracy_state = model.state_dict() 

        per_epoch_accuracy.append(accuracy) 

        logging.info(f"max accuracy reached: {hyper_parameters['max_accuracy']}")

        n_epochs += 1 
    
    logging.info("training done... saving the model and plots") 

    save_dir = "./model/trained_model"

    session_save(max_accuracy_state, save_dir, per_epoch_loss, per_epoch_accuracy, hyper_parameters) 

    logging.info("execution complete")
