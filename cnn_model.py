import torch.nn as nn 
import torch 


class Model(nn.Module): 
    
    def __init__(self, num_channels, kernel_size, pool_size) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size = kernel_size, stride = 1, padding = 'same') 
        self.act1 = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(pool_size, stride = 1)

        self.conv2 = nn.Conv2d(num_channels,num_channels, kernel_size = kernel_size, stride = 1, padding = 'same') 
        self.act2 = nn.ReLU() 
        self.pool2 = nn.MaxPool2d(pool_size, stride = 1) 
        
        self.flatten = nn.Flatten() 

        self.fc1 = nn.Linear(num_channels*26*26, 120) 
        self.act_fc1 = nn.ReLU() 
        self.fc2 = nn.Linear(120, 84) 
        self.act_fc2 = nn.ReLU() 
        self.fc3 = nn.Linear(84, 10) 

        self.sm = nn.Softmax(dim = 1) 

    def forward(self, x): 
        x = self.pool1(self.act1(self.conv1(x))) 

        x = self.pool2(self.act2(self.conv2(x)))

        x = self.flatten(x)

        x = self.act_fc1(self.fc1(x)) 
        x = self.act_fc2(self.fc2(x)) 

        x = self.fc3(x) 

        return x
    
    def predict(self, x): 

        output = self.forward(x) 

        return self.sm(output) 
    
    def model_train(self, loss_func, optimizer, train_data_loader): 

        total_training_loss = 0 
        
        for i, data in enumerate(train_data_loader): 

            inputs, labels = data 

            optimizer.zero_grad() 

            outputs = self.forward(inputs) 

            training_loss = loss_func(outputs, labels) 

            training_loss.backward() 

            optimizer.step() 

            total_training_loss += training_loss.item() 

        return total_training_loss
    

    def model_test(self, test_loader): 

        classified = 0 
        total_inputs = 0 
        for i, data in enumerate(test_loader): 

            inputs, labels = data 

            outputs = self.predict(inputs) 

            _, predictions = torch.max(outputs, dim = 1) 

            classified += sum(map(int, torch.eq(labels, predictions)))

            total_inputs += len(labels) 

        accuracy = float(classified / total_inputs) 

        return accuracy 