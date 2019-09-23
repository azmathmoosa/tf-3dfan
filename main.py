"""
68-pt Facial Landmark Extractor
@author: azmath
"""

import numpy as np 
import tensorflow as tf 
import cv2 


import data_gen
from utils import get_landmarks
from config import *
import model 

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    """MAIN"""
    est_config = tf.estimator.RunConfig(
        save_checkpoints_steps = 5000,  # Save checkpoints every 100 steps.
        keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
        save_summary_steps=100,        
    )

    exporter = tf.estimator.BestExporter(
        serving_input_receiver_fn=model._serving_input_receiver_fn,
        exports_to_keep=5
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=model._train_input_fn,
        max_steps=TRAIN_MAX_STEPS
    )


    eval_spec = tf.estimator.EvalSpec(
        input_fn=model._eval_input_fn,
        steps=1000,
        throttle_secs=15*60,
        exporters=exporter
    )

    estimator = tf.estimator.Estimator(
        model_fn=model.cnn_model_fn,
        model_dir=MODEL_DIR,
        config=est_config
    )

    # Choose mode between Train, Evaluate and Predict
    mode_dict = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }

    mode = 'export'#mode_dict['export']

    if mode == mode_dict['train']:
        tf.estimator.train_and_evaluate(
            estimator,
            train_spec,
            eval_spec
        )

    elif mode == mode_dict['eval']:
        evaluation = estimator.evaluate(input_fn=model._eval_input_fn)
        tf.print(evaluation)

    elif mode == "export":
        estimator.export_saved_model('%s/saved_model'%EXPORT_DIR, model._serving_input_receiver_fn)


    elif mode == mode_dict['predict']:
        predictions = estimator.predict(input_fn=model._predict_input_fn, yield_single_examples=False)        
        for result in predictions:
            filename  = result['name'][0].decode('ASCII')
            print("Evaluating %s"%filename)
            img = result['image'] #cv2.imread(filename)
            heatmaps = result['heatmap']

            pts = get_landmarks(heatmaps[0][-1])
            print("Landmark Points"%pts)

            for i, heatmap in enumerate(heatmaps):
                heatmap  = np.sum(heatmap[0], axis=2)
                # heatmap = (heatmap / -255).astype(np.uint8)
                heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
                heatmap = cv2.resize(heatmap, (256, 256))
                cv2.imshow("%d"%i, heatmap)

            for pt in pts:
                cv2.circle(img[0], (int(pt[1]), int(pt[0])), 2, (0, 255, 0), -1, cv2.LINE_AA)
                      
            cv2.imshow('result', img[0])
            cv2.waitKey(0)


if __name__ == "__main__":
    tf.app.run(main=main)