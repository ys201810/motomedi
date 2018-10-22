# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import cv2
import predict
import numpy as np
import tensorflow as tf
import keras

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'PNG', 'JPG', 'GIF'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MODEL_DIR'] = './model/'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    return render_template('index.html')

@app.route('/class_sf_rand')
def class_sf_rand():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    model_list = os.listdir(app.config['MODEL_DIR'])
    return render_template('classification_side_fork.html', model_name = 'classification_side_fork')

@app.route('/class_ff_rand')
def class_ff_rand():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    model_list = os.listdir(app.config['MODEL_DIR'])
    return render_template('classification_front_fork.html', model_name = 'classification_front_fork')


@app.route('/send', methods=['GET', 'POST'])
def send():
    img_url, result, feature, model_list = aaa()
    return render_template('classification_side_fork.html', img_url=img_url, result=result, confidences=feature , model_list = model_list)

def aaa():
    model_list = os.listdir(app.config['MODEL_DIR'])

    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            print(filename)

            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resize_img = cv2.resize(img, (400, 300))
            resize_name = os.path.join(app.config['UPLOAD_FOLDER'],
                                     filename.split('.')[0] + '_resize34.' + filename.split('.')[1])

            cv2.imwrite(resize_name, resize_img)

            img_url = '/uploads/' + filename.split('.')[0] + '_resize34.' + filename.split('.')[1]
            use_model = app.config['MODEL_DIR'] + request.form['models']

            result, feature = predict.predict(resize_name, use_model)

            feature = np.round(feature, 3)
            feature = feature * 100

            return img_url, result, feature, model_list
        else:
            return  ''' <p>許可されていない拡張子です。</p> '''
    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        if username == 'admin':
            session['username'] = request.form['username']
            return redirect(url_for('index'))
        else:
            return '''<p>ユーザー名が違います</p>'''
    return '''
        <form action="" method="post">
            <p><input type="text" name="username">
            <p><input type="submit" value="Login">
        </form>
    '''

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.debug = True
    app.run()
