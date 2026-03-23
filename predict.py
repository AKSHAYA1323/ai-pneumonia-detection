import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/pneumonia_model.keras")

IMG_SIZE = 224

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print(f"PNEUMONIA detected (Confidence: {prediction:.4f})")
    else:
        print(f"NORMAL (Confidence: {prediction:.4f})")

# Test
predict_image("dataset/test/PNEUMONIA/person1_virus_6.jpeg")