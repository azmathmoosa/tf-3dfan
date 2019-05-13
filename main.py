"""
68-pt Facial Landmark Extractor
@author: azmath
"""

import numpy as np 
import tensorflow as tf 
import cv2 
from fan import FAN
import data_gen

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3
IMG_DIM = IMG_WIDTH
data_gen.IMG_DIM = IMG_DIM
HG_STACK = 1

tf.logging.set_verbosity(tf.logging.INFO)

def _parse_function(face, gtmap, filename):
    return {"x": face, "name": filename }, gtmap

def input_fn(purpose='train', batch_size=1, num_epochs=None, shuffle=True):
    """
    Input function for tf estimator
    """
    dataset = tf.data.Dataset.from_generator(lambda: data_gen.generator(purpose, augment=True),
                                output_types=(tf.float32, tf.float32, tf.string),
                                output_shapes=((IMG_DIM, IMG_DIM, 3), (2, 64, 64, 68), tf.TensorShape([])))

    dataset = dataset.map(_parse_function)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10)
    if batch_size != 1:
        dataset = dataset.batch(batch_size)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.prefetch(buffer_size=100)
    
    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    feature, label = iterator.get_next()
    return feature, label


def _train_input_fn():
    """Function for training."""
    return input_fn(
        purpose='train',        
        batch_size=10, 
        num_epochs=50, 
        shuffle=True)

def _eval_input_fn():
    """Function for evaluating."""
    return input_fn(
        purpose='eval',
        batch_size=2,
        num_epochs=1,
        shuffle=True)

def _predict_input_fn():
    """Function for predicting."""
    return input_fn(
        purpose='eval',
        batch_size=2,
        num_epochs=1,
        shuffle=True)

def _serving_input_receiver_fn():
    """An input receiver that expects an image input"""
    image = tf.placeholder(
        dtype=tf.uint8,
        shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL],
        name="image_tensor"
    )
    img_path = tf.placeholder(
        dtype=tf.string,
        shape=tf.TensorShape([]),
        name="img_path"
    )
    receiver_tensor = {'image': image, 'name': img_path}
    feature = {
        'x': tf.reshape(image, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL]),
        'name': img_path
        }
    return tf.estimator.export.ServingInputReceiver(feature, receiver_tensor)

def cnn_model_fn(features, labels, mode):

    inputs = tf.to_float(features['x'], name="input_to_float")

    heatmaps = FAN(HG_STACK)(inputs)

    # # Make prediction for PREDICTION mode.
    predictions_dict = {
        "heatmap": heatmaps
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        

        predictions_dict = {
            "name": features['name'],
            "heatmap": tf.convert_to_tensor(heatmaps),
            "image": inputs
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)


    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
    gttensor = tf.math.reduce_sum(labels_tensor, axis=4, keepdims=True)

    heatmaps = tf.stack(heatmaps, axis=1)
    outtensor = tf.math.reduce_sum(heatmaps, axis=4, keepdims=True)

    tf.summary.image("input", inputs)
    tf.summary.image("outputs", outtensor[:,-1])
    tf.summary.image("gtmap", gttensor[:,-1])

    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=heatmaps, labels=labels_tensor), name= 'cross_entropy_loss')
    loss = tf.losses.mean_squared_error(
        predictions=heatmaps, labels=labels_tensor
    )
    # loss = (tf.nn.l2_loss(heatmaps - labels_tensor))/2

    # loss = tf.losses.mean_squared_error(
    #     labels=labels_tensor,
    #     predictions=heatmaps,
    #     weights=1.
    # )

    tf.summary.scalar("htval", heatmaps[0,0,0,0,0])
    tf.summary.scalar("gtval", labels_tensor[0,0,0,0,0])
    tf.summary.scalar("loss", loss)
        
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=0.001
        )
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op 
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "MSE": tf.metrics.mean_squared_error(
                labels=labels_tensor,
                predictions=heatmaps 
            ),
            "CrossEntropy": tf.metrics.mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=heatmaps, labels=labels_tensor)
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )    

def main(unused_argv):
    """MAIN"""
    est_config = tf.estimator.RunConfig(
        save_checkpoints_steps = 5000,  # Save checkpoints every 100 steps.
        keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
        save_summary_steps=100,        
    )

    exporter = tf.estimator.BestExporter(
        serving_input_receiver_fn=_serving_input_receiver_fn,
        exports_to_keep=5
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=_train_input_fn,
        max_steps=1000000
    )


    eval_spec = tf.estimator.EvalSpec(
        input_fn=_eval_input_fn,
        steps=100,
        throttle_secs=15*60,
        exporters=exporter
    )

    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="./train-tf-fan-mse-5",
        config=est_config
    )

    # Choose mode between Train, Evaluate and Predict
    mode_dict = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }

    mode = mode_dict['train']

    if mode == mode_dict['train']:
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )
        # estimator.train(
        #     input_fn=_train_input_fn,
        #     steps=2000000
        # )
        # estimator.export_saved_model(
        #     export_dir_base="./exported",
        #     serving_input_receiver_fn=_serving_input_receiver_fn
        # )

    if mode == mode_dict['eval']:
        evaluation = estimator.evaluate(input_fn=_eval_input_fn)
        tf.print(evaluation)

    else:
        predictions = estimator.predict(input_fn=_eval_input_fn)
        for _, result in enumerate(predictions):
            filename  = result['name'].decode('ASCII')
            print(filename)
            img = result['image'] #cv2.imread(filename)
            heatmaps = result['heatmap']

            pts = get_landmarks(heatmaps[1])
            print(pts)

            for i, heatmap in enumerate(heatmaps):
                heatmap  = np.sum(heatmap, axis=2)
                # heatmap = (heatmap / -255).astype(np.uint8)
                heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
                heatmap = cv2.resize(heatmap, (256, 256))
                cv2.imshow("%d"%i, heatmap)


            # print(heatmap)
            for pt in pts:
                cv2.circle(img, (int(pt[1]), int(pt[0])), 2, (0, 255, 0), -1, cv2.LINE_AA)

                # cv2.imshow("map%d"%i, heatmap)

            # marks = np.squeeze(np.reshape(result['logits'], (2,68))) 
            # sz = IMG_WIDTH*2
            # sizes = np.array([sz, sz]).reshape(-1, 1)
            # print(sizes)
            # markst = np.transpose(marks * sizes)
            # print(markst)
            # img = cv2.resize(img, (sz, sz))
            # for mark in markst:
            #     cv2.circle(img, (int(mark[0]), int(
            #         mark[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)                        
            cv2.imshow('result', img)
            cv2.waitKey(0)

def get_landmarks(heatmaps):
    pts = []
    heatmaps = np.moveaxis(heatmaps, -1, 0)
    for i,heatmap in enumerate(heatmaps):   
        
        heatmap = cv2.resize(heatmap, (IMG_DIM, IMG_DIM))   
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
        # if 28 < i < 35:
        #     cv2.imshow("pt%d"%i, heatmap)
        

        pt = np.unravel_index(heatmap.argmax(), heatmap.shape)
        pt = pt[0], pt[1]
        pts.append(pt)
    
    return pts

if __name__ == "__main__":
    tf.app.run(main=main)