"""
68-pt Facial Landmark Extractor
@author: azmath
"""

import numpy as np 
import tensorflow as tf 
import cv2 
import fan
import data_gen

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3
IMG_DIM = IMG_WIDTH
data_gen.IMG_DIM = IMG_DIM

tf.logging.set_verbosity(tf.logging.INFO)

def _parse_function(face, gtmap, filename):
    return {"x": face, "name": filename }, gtmap

def input_fn(purpose='train', batch_size=1, num_epochs=None, shuffle=True):
    """
    Input function for tf estimator
    """
    dataset = tf.data.Dataset.from_generator(lambda: data_gen.generator(purpose),
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
        num_epochs=3000, 
        shuffle=True)

def _eval_input_fn():
    """Function for evaluating."""
    return input_fn(
        purpose='eval',
        batch_size=2,
        num_epochs=1,
        shuffle=False)

def _predict_input_fn():
    """Function for predicting."""
    return input_fn(
        purpose='eval',
        batch_size=2,
        num_epochs=1,
        shuffle=False)

def _serving_input_receiver_fn():
    """An input receiver that expects an image input"""
    image = tf.placeholder(
        dtype=tf.uint8,
        shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL],
        name="image_tensor"
    )
    receiver_tensor = {'image': image}
    feature = {
        'x': tf.reshape(image, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        }
    return tf.estimator.export.ServingInputReceiver(feature, receiver_tensor)

def cnn_model_fn(features, labels, mode):

    inputs = tf.to_float(features['x'], name="input_to_float")

    heatmaps = FAN(2)(inputs)

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

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=heatmaps, labels=labels_tensor), name= 'cross_entropy_loss')

    # loss = tf.losses.mean_squared_error(
    #     labels=labels_tensor,
    #     predictions=heatmaps,
    #     weights=1.
    # )

    tf.summary.scalar("loss", loss)
        
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.0001
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
            "MSE": tf.metrics.root_mean_squared_error(
                labels=labels_tensor,
                predictions=heatmaps 
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
        save_checkpoints_steps = 1000,  # Save checkpoints every 100 steps.
        keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
        save_summary_steps=100,        
    )

    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="./train-fan-ce",
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
        estimator.train(
            input_fn=_train_input_fn,
            steps=200000
        )
        estimator.export_saved_model(
            export_dir_base="./exported",
            serving_input_receiver_fn=_serving_input_receiver_fn
        )

    if mode == mode_dict['eval']:
        evaluation = estimator.evaluate(input_fn=_eval_input_fn)
        tf.print(evaluation)

    else:
        predictions = estimator.predict(input_fn=_predict_input_fn)
        for _, result in enumerate(predictions):
            filename  = result['name'].decode('ASCII')
            print(filename)
            img = result['image'] #cv2.imread(filename)
            heatmaps = result['heatmap']

            print(heatmaps)
            for i, heatmap in enumerate(heatmaps):
                heatmap  = np.sum(heatmap, axis=2)
                print(np.unravel_index(np.argmax(heatmap), np.array(heatmap).shape))
                heatmap = (heatmap * 255).astype(np.uint8)

                cv2.imshow("map%d"%i, heatmap)

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

if __name__ == "__main__":
    tf.app.run(main=main)