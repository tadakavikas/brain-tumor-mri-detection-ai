import numpy as np
import tensorflow as tf
from PIL import Image
import json
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("models/brain_tumor_model.h5")

with open("models/class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

def preprocess_for_model(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image):
    img = preprocess_for_model(image)

    prediction = model.predict(img, verbose=0)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    predicted_label = labels[class_index]

    heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name="conv2d_2", pred_index=class_index)
    overlay = overlay_heatmap_on_image(image, heatmap)

    return predicted_label, confidence, overlay