import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


application = Flask(__name__)
model = pickle.load(open('house-price.pkl','rb')) 

@application.route('/')
def home():
  
    return render_template("index.html")
  
@application.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted Price of House for given area  is : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
     
