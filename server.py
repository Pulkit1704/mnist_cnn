from flask import Flask, jsonify, request, render_template
from lib.utils import make_predictions

app = Flask(__name__) 

@app.route('/') 
def root(): 
    return render_template("index.html")

@app.route('/predict', methods=['POST']) 
def predictions_endpoint(): 
   
    if request.method == 'POST': 
        
        file = request.data

        predicted_class = make_predictions(file)
    
    return jsonify(predicted_class.item()) 


if __name__ == "__main__": 

    host = "127.0.0.1"
    port_number = 8080 

    app.run(host, port_number)