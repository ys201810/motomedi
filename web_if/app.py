# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import cv2
import predict
import numpy as np
from od_predict import YOLO
from PIL import Image

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

@app.route('/class_shashu_rand')
def class_shashu_rand():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    return render_template('classification_shashu.html', model_name = 'classification_shashu')

@app.route('/class_sf_rand')
def class_sf_rand():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    return render_template('classification_side_fork.html', model_name = 'classification_side_fork')

@app.route('/class_ff_rand')
def class_ff_rand():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    return render_template('classification_front_fork.html', model_name = 'classification_front_fork')

@app.route('/detec_sf_rand')
def detec_sf_rand():
    """ This is first access point. Get the model list from model directory and send for html's select box."""
    return render_template('detection_side_fork.html', model_name = 'detection_side_fork')

@app.route('/send_shashu', methods=['GET', 'POST'])
def send_shashu():
    use_model = './model/cnn_model_weights_type.hdf5'
    img_url, first_idx, second_idx, third_idx, feature = class_inference(use_model, 'shashu')
    shashuh_true_list = ['bolt','cb1300','gsx_r1000','gsx_r1000r','gsx_s1000','gsx_s1000f','mt_07','mt_09','ninja','sv650','tmax','trecer900','v_strom_1000','v_strom_650','v_strom_650xt','yamaha__sr400','yzf_r1','yzf_r25','zrx1200']
    first_shashu = shashuh_true_list[first_idx]
    second_shashu = shashuh_true_list[second_idx]
    third_shashu = shashuh_true_list[third_idx]
    return render_template('classification_shashu.html', img_url=img_url, first_shashu=first_shashu, second_shashu=second_shashu, third_shashu=third_shashu,
                           confidences=feature , model_name = 'classification_shashu')

@app.route('/send_sf', methods=['GET', 'POST'])
def send_sf():
    use_model = './model/cnn_model_weights_side.hdf5'
    img_url, first_idx, second_idx, third_idx, feature = class_inference(use_model, 'fork')
    return render_template('classification_side_fork.html', img_url=img_url, result=first_idx,
                           confidences=feature , model_name = 'classification_side_fork')

@app.route('/send_ff', methods=['GET', 'POST'])
def send_ff():
    use_model = './model/cnn_model_weights_ff.hdf5'
    img_url, first_idx, second_idx, third_idx, feature = class_inference(use_model, 'fork')
    return render_template('classification_front_fork.html', img_url=img_url, result=first_idx,
                           confidences=feature , model_name = 'classification_front_fork')

@app.route('/send_sf_det', methods=['GET', 'POST'])
def send_sf_det():
    use_model = './model/side_od_trained_weights_final.h5'
    img_url, result_url = det_inference(use_model)
    return render_template('detection_side_fork.html', img_url=img_url, result_url=result_url,
                           model_name = 'detection_side_fork')

def class_inference(use_model, process_kbn):
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
            use_model = use_model

            if process_kbn == 'fork':
                num_class = 2
            elif process_kbn == 'shashu':
                num_class = 19

            first_idx, second_idx, third_idx, feature = predict.predict(resize_name, use_model, num_class)

            feature = np.round(feature, 3)
            feature = feature * 100

            return img_url, first_idx, second_idx, third_idx, feature
        else:
            return  ''' <p>許可されていない拡張子です。</p> '''
    else:
        return redirect(url_for('index'))

def det_inference(use_model):
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            print(filename)

            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resize_img = cv2.resize(img, (416, 320))
            resize_name = os.path.join(app.config['UPLOAD_FOLDER'],
                                     filename.split('.')[0] + '_resize34.' + filename.split('.')[1])
            cv2.imwrite(resize_name, resize_img)
            resize_img_url = '/uploads/' + filename.split('.')[0] + '_resize34.' + filename.split('.')[1]

            image = Image.open(resize_name)
            yolo = YOLO()
            r_image = yolo.detect_image(image)

            result_name = os.path.join(app.config['UPLOAD_FOLDER'],
                                     filename.split('.')[0] + '_result_det.' + filename.split('.')[1])
            r_image.save(result_name, quality=100, optimize=True)
            # cv2.imwrite(result_name, r_image)
            result_img_url = '/uploads/' + filename.split('.')[0] + '_result_det.' + filename.split('.')[1]

            return resize_img_url, result_img_url


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
    app.run(host='0.0.0.0', port=5000)
