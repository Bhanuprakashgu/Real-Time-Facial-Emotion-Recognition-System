from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.image import rgb_to_grayscale

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model_filter.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

USERNAME = 'admin@gmail.com'
PASSWORD = 'admin'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == USERNAME and password == PASSWORD:
            session['user_id'] = username  
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/input')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the image and convert to grayscale
        img = image.load_img(filepath, target_size=(48, 48))  # Resize to model's expected input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = rgb_to_grayscale(img_array)  # Convert to grayscale

        # Normalize the pixel values to [0,1]
        img_array /= 255.0

        # Predict the class using the model
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)  # Get the index of the highest score

        # Define the emotion classes
        emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']
        result = emotion_classes[class_index]  # Get the emotion based on the predicted index

        return render_template('result.html', predicted_class=result, image_file=filepath)

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/chart')
def chart():
    return render_template('chart.html')

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'
    app.run()
