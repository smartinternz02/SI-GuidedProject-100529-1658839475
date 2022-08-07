import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
global sess

global graph
graph=tf.compat.v1.get_default_graph()



model = load_model("C:\\Fertilizers RS For Disease Prediction\\fruit_pathon.h5")
model1=load_model("C:\\Fertilizers RS For Disease Prediction\\veg_pathon.h5")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])		

def predict():
    if request.method == 'POST':

        f = request.files['image']


        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        plant=request.form['plant']
        print(plant)

        if(plant=="vegetable"):
            img = image.load_img(file_path, target_size=(128,128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            preds = model1.predict(x)
            preds = np.argmax(preds)
            print(preds)
            index=['Pepper_BS','Pepper_H','Potato_EB','Potato_LB','Potato_H','Tomato_BS','Tomato_blight','Tomato_mold','Tomato_H']
            print(index[preds])
            df=pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[preds]['caution'])
        else:
            img = image.load_img(file_path, target_size=(64,64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            preds = np.argmax(preds)
            print(preds)
            index=['Apple_BR','Apple_H','Corn_NLB','Corn_H','Peach_BS','Peach_H']
            print(index[preds])
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds]['caution'])
            
        
        return df.iloc[preds]['caution']

        
 
        
     
if __name__ == "__main__":
    app.run(debug=False)
    