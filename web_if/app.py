# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import cv2
import predict
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
    model_list = os.listdir(app.config['MODEL_DIR'])
    return render_template('index.html', model_list = model_list)

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


@app.route('/send', methods=['GET', 'POST'])
def send():
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
            result = predict.predict(resize_name, use_model)
            return render_template('index.html', img_url=img_url, result=result, model_list = model_list)
        else:
            return  ''' <p>許可されていない拡張子です。</p> '''
    else:
        return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.debug = True
    app.run()
