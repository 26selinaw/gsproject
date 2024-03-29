# recognizer/Users/26selinaw/Desktop/run.py
import cv2
import numpy as np
import mediapipe as mp
import sys
import os
import io
import tensorflow as tf
import tensorflow.compat.v1 as tf
from numpy import array
from tensorflow.keras.models import load_model
from PIL import Image

# c:disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# m:initialize the webcam
cap = cv2.VideoCapture(0)

# c:loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("image-classification-tensorflow/logs/trained_labels.txt")]

# c:unpersists graph from file; load model
with tf.io.gfile.GFile("image-classification-tensorflow/logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

while True:
    # m:read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # m:flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #framesh = ''.join(str(e) for e in framergb)

    im = Image.fromarray(framergb)
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        frames = output.getvalue()
    
    with tf.compat.v1.Session() as sess:
        # c:feed the image_data
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': frames})
                
    # m:print(prediction)
    classID = np.argmax(predictions)
    className = label_lines[classID]

    # m:show the prediction on the frame
    cv2.putText(frame, className, (550, 320), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (230,230,250), 2, cv2.LINE_AA)

    # m:show the final output
    cv2.imshow("Selina", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    print("finish frame")

# m: release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
