from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image

app = Flask(__name__)


model = tf.keras.models.load_model('digit_finder.h5')


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        file = request.files['image']
        
        filename = 'temp.jpg'
        file.save(filename)
        
        imgpath = Image.open(filename)
        img = cv.imread(filename)
        resized = cv.resize(cv.cvtColor(img,cv.COLOR_BGR2GRAY),(28,28),interpolation = cv.INTER_AREA)
        newimg = np.array(tf.keras.utils.normalize(resized,axis= 1)).reshape(-1,28,28,1)
        predictions = model.predict(newimg)
    
        pred = np.argmax(predictions)
    
        return render_template('success.html', prediction=pred)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)