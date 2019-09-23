import tensorflow as tf
import os 


chkpoint = "876669"
chkpoint_dir = "train-tf-fan-mse-5/"

meta_path = '%s/model.ckpt-%s.meta'%(chkpoint_dir, chkpoint) # Your .meta file
output_node_names = ['heatmap:0']    # Output nodes

if not os.path.exists("exported/"):
    os.makedirs("exported/")

with tf.Session() as sess:

    print("Loading meta from %s"%meta_path)
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(chkpoint_dir))

    with open("nodenames.txt", "w") as nf:
        for n in tf.get_default_graph().as_graph_def().node:
            nf.write(n.name+"\n")
    
    

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph

    with open('exported/output_graph-%s.pb'%chkpoint, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())