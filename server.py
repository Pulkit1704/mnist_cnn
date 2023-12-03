from io import BytesIO
from flask import Flask, jsonify, request, render_template
from lib.image_preprocess import transform_image
import base64 
from PIL import Image 

import matplotlib.pyplot as plt 

app = Flask(__name__) 

@app.route('/') 
def hello(): 
    return render_template("index.html")

@app.route('/predict', methods=['POST']) 
def make_prediction(): 
   
    if request.method == 'POST': 
        
        file = request.data
        data = file.split(b',')[-1]
        image_data = base64.decodebytes(data) 

        pil_image = Image.open(BytesIO(image_data))

        # image_tensor = torch.from_numpy(np.array(image))

        # print(image_tensor.shape) 
        
        processed_image = transform_image(pil_image) 
        plt.imshow(processed_image.numpy()[3])
        plt.savefig("./sample.png") 
        return jsonify({'class': processed_image.shape})
