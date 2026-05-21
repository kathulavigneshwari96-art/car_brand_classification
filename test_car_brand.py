# -----------------------------
# TEST CAR BRAND MODEL
# -----------------------------

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("car_brand_model.h5")

# Class names (same order as folder names)
class_names = ["Audi", "BMW", "Honda", "Tesla", "Toyota"]

# -----------------------------
# CHANGE FILENAME HERE
# -----------------------------
img_path = r"Audi/Audi20.jpg"
 # <-- Put your test file name

# -----------------------------
# LOAD + PREDICT
# -----------------------------
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred = model.predict(img_array)
predicted_class = np.argmax(pred)
print("Predicted Brand:", class_names[predicted_class])

# SHOW IMAGE
plt.imshow(img)
plt.title(f"Predicted: {class_names[predicted_class]}")
plt.axis("off")
plt.show()