# -*- coding: utf-8 -*- 
import configparser
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from objectdetection.yolov3 import yolo_body, yolo_loss, preprocess_true_boxes
from keras.models import Model
import common.tensorboard_conf
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from objectdetection.yolov3_utils import get_random_data


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def get_classes(classes_path):
    with open(classes_path, 'r') as classes_file:
        class_names = classes_file.readlines()
    class_names = [each_class.strip() for each_class in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as anchors_file:
        anchors = anchors_file.readline()
    anchors = [float(anchor) for anchor in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)




def create_model(input_shape, anchors, num_classes, freeze_body, weights_path, load_pretrained):
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # h//で割り算の商だけ取得。tf.Tensorのshapeとして、(?, h, w, anchors, result[each class_confidence, x, y, w, h, loc_conf])を3つ作成
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes + 5)) for l in range(3)]
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        # by_name = True:同じ名前のレイヤーにのみ読み込み skip_mismatch
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i ==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def main():
    config_file = '../conf/config_od.ini'
    config = load_config(config_file)

    # set config
    image_height = int(config.get('image_info', 'image_height'))
    image_width = int(config.get('image_info', 'image_width'))
    image_channel_dim = int(config.get('image_info', 'image_channel_dim'))
    save_base_dir = config.get('other_info', 'save_dir')
    annotation_path = config.get('label_info', 'annotation_path')
    anchors_path = config.get('label_info', 'anchors_path')
    classes_path = config.get('label_info', 'classes_path')
    pretrain_model_path = config.get('train_info', 'pre_train_model')
    batch_size = int(config.get('train_info','batch_size'))
    epochs = int(config.get('train_info', 'epochs'))

    now = datetime.datetime.now()
    experiment_id = now.strftime('%Y%m%d_%H%M')
    save_dir = save_base_dir + experiment_id
    model_dir = save_dir + '/checkpoint/'
    log_dir = save_dir + '/tensorboard/'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (image_height, image_width)

    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=pretrain_model_path, load_pretrained=True)

    write_graph = True
    histogram_freq = 0
    tensorboard = tensorboard_conf.TrainValTensorBoard(log_dir=log_dir, write_graph=write_graph, histogram_freq=histogram_freq)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as inf:
        lines = inf.readlines()
    np.random.seed(1)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to own dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr = 1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all of the layers.')
        batch_size = batch_size
        print('Train on {} samples, val on {} samples, with batch_size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch = max(1, num_train // batch_size),
            validation_data = data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps = max(1, num_val // batch_size),
            epochs=epochs,
            initial_epoch=0,
            callbacks=[tensorboard, checkpoint]
        )
        model.save_weights(model_dir + 'trained_weight_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr =1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all of the layers.')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs = epochs,
            initial_epoch = epochs,
            callbacks=[tensorboard, checkpoint, reduce_lr, early_stopping]
        )
        model.save_weights(model_dir, + 'trained_weights_final.h5')

if __name__ == '__main__':
    main()