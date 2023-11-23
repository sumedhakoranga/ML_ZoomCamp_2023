import numpy as np
import os
import tensorflow.lite as tflite
from urllib import request
from PIL import Image
from io import BytesIO

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    img = prepare_image(
        download_image(url),
        (150, 150)
    )

    img = np.asarray(img).astype('float32')/255

    interpreter.set_tensor(input_index, [img])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return float(preds[0][0])


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)

    return result