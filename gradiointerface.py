import tensorflow as tf
import gradio as gr
import numpy as np
from keras.preprocessing import image

def classify2(photo):
    classmodel = tf.keras.models.load_model('cnnMod1Catdog100')
    test_image = image.img_to_array(photo)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classmodel.predict(test_image)
    if result[0][0] == 1:
        prediction = 'This is a picture of a dog!'
    else:
        prediction = 'This is a picture of a cat!'
    return prediction



inputclassify = gr.inputs.Image(source="upload", shape=([64,64]), type="pil", label = "Upload a image of a Cat or Dog to be classified.")
classinter1 = gr.Interface(fn=classify2, inputs=[inputclassify], outputs="text")
classinter1.launch()