import tensorflow as tf 

from fan import FAN
import data_gen
from config import * 

def _parse_function(face, gtmap, filename):
    return {"x": face, "name": filename }, gtmap

def input_fn(purpose='train', batch_size=1, num_epochs=None, shuffle=True):
    """
    Input function for tf estimator
    """
    dataset = tf.data.Dataset.from_generator(lambda: data_gen.generator(purpose, augment=True),
                                output_types=(tf.float32, tf.float32, tf.string),
                                output_shapes=((IMG_DIM, IMG_DIM, 3), (HG_STACK, 64, 64, 68), tf.TensorShape([])))

    dataset = dataset.map(_parse_function)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10)
    
    dataset = dataset.batch(batch_size)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.prefetch(buffer_size=50)
    
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
        batch_size=TRAIN_BATCH_SIZE, 
        num_epochs=TRAIN_EPOCHS, 
        shuffle=True)

def _eval_input_fn():
    """Function for evaluating."""
    return input_fn(
        purpose='eval',
        batch_size=EVAL_BATCH_SIZE,
        num_epochs=1,
        shuffle=True)

def _predict_input_fn():
    """Function for predicting."""
    return input_fn(
        purpose='eval',
        batch_size=1,
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
            "heatmap": tf.convert_to_tensor(heatmaps), #if HG_STACK > 1 else tf.transpose(tf.convert_to_tensor(heatmaps),[1,0,2,3,4]),
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

    wt_thres = 0
    wt_cond = tf.greater(labels_tensor, tf.ones(tf.shape(labels_tensor))*wt_thres)
    wt_mask = tf.where(wt_cond, tf.ones(tf.shape(labels_tensor))*10, tf.ones(tf.shape(labels_tensor)))

    loss = tf.losses.mean_squared_error(
        predictions=heatmaps, labels=labels_tensor, weights=wt_mask
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
            learning_rate=0.00001
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
