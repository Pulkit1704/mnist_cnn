from flask import Flask, jsonify, request, render_template
from lib.utils import transform_image, make_predictions

app = Flask(__name__) 

@app.route('/') 
def hello(): 
    return render_template("index.html")

@app.route('/predict', methods=['POST']) 
def make_prediction(): 
   
    if request.method == 'POST': 
        
        file = request.data
        processed_image = transform_image(file)

        predicted_class = make_predictions(processed_image)
    
    return jsonify(predicted_class.item()) 
