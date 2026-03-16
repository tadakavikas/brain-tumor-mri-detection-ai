import numpy as np
import tensorflow as tf
import cv2

IMG_SIZE = (224, 224)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4):
    original_image = original_image.resize(IMG_SIZE)
    original_array = np.array(original_image)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original_array, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed