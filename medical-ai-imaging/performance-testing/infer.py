# Created by Scalers AI for Dell Inc.

import tensorflow as tf
import time
import fire
import cv2
import numpy as np

def load_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

def infer(
    model: str,
    num_infer: int = 100,
    image_path: str = "xray.png",
    input_layer: str = 'x:0',
    output_layer: str = 'model_7/dense_15/Softmax:0',
    ):
    frozen_graph_path = model
    image = load_img(image_path)
    
    # Load the graph definition
    with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.import_graph_def(graph_def, name='')
        input_tensor = sess.graph.get_tensor_by_name(input_layer)
        output_tensor = sess.graph.get_tensor_by_name(output_layer)                 
                                            
        image = np.expand_dims(image, axis=-1)
        
        # warm up
        for _ in range(10):
            output = sess.run(output_tensor, feed_dict={input_tensor: image})

        start_time = time.time()
        for _ in range(num_infer):
            output = sess.run(output_tensor, feed_dict={input_tensor: image})
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time/num_infer} sec")
    
if __name__ == "__main__":
    fire.Fire(infer)