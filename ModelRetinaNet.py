import main
import numpy as np
import gradio as gr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

image = tf.keras.Input(shape=[None, None, 3], name="image")

def predict(input_img):
    # Assuming main.predict_model returns a dictionary
    detection_result = main.predict_model(input_img)
    print(detection_result[1])
    # Visualize detections and return the image path
    return detection_result[0]

iface = gr.Interface(
    fn=predict,
    inputs="image",
    outputs="image"  # Now outputting the image path
)

# Launch the Gradio interface
iface.launch()