import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('KNN-model.pkl','rb')) 

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])

def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    PClass = float(request.args.get('PClass'))
    Sex = float(request.args.get('Sex'))
    Age = float(request.args.get('Age'))
    Sibsp = float(request.args.get('Sibsp'))
    Parch = float(request.args.get('Parch'))
    Fare = float(request.args.get('Fare'))
    
    prediction = model.predict([[PClass,Sex,Age,Sibsp,Parch,Fare]])
    if prediction == [1]:
        text = "Person was able to survive"
    else:
        text = "Person was not able to survive"
    return render_template('index.html', prediction_text='KNN model to predict if the person survived in titanic incident or not: {}'.format(text))
    
        
    
if __name__=="__main__":
    app.run(debug=True)