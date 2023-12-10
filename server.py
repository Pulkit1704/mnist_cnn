from flask import Flask, jsonify, request, render_template
from lib.image_preprocess import transform_image
from model.cnn_model import Model 
import torch 

app = Flask(__name__) 

@app.route('/') 
def hello(): 
    return render_template("index.html")

@app.route('/predict', methods=['POST']) 
def make_prediction(): 
   
    if request.method == 'POST': 
        
        file = request.data
        processed_image = transform_image(file)

        model = Model(10, 3, 2) 
        model.load_state_dict(torch.load("model/trained_model/mnist_trained.pth"))

        predictions_tensor = model.predict(torch.reshape(processed_image[3], [1, 1, 28, 28]))

        _, predicted_class = torch.max(predictions_tensor, dim=1) 
    
    return jsonify(predicted_class.item()) 
