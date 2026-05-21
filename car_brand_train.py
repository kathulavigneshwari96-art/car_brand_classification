# -----------------------------
# CAR BRAND IMAGE CLASSIFIER
# TRAINING SCRIPT
# -----------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

# -----------------------------
# LOAD DATASET
# -----------------------------
data_path = "."

# Image augmentation + rescaling
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2
)

# Training set
train_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=16,
    subset="training",
    class_mode="categorical"
)

# Validation set
val_data = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=16,
    subset="validation",
    class_mode="categorical"
)

# Number of classes detected
num_classes = train_data.num_classes

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"],
 )


# -----------------------------
# TRAIN THE MODEL
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# -----------------------------
# SAVE THE MODEL
# -----------------------------
model.save("car_brand_model.h5")
print("Model saved as car_brand_model.h5")

# -----------------------------
# ACCURACY GRAPH
# -----------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

# -----------------------------
# LOSS GRAPH
# -----------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

# Get true labels
true_labels = val_data.classes

# Predict probabilities
pred_prob = model.predict(val_data)

# Convert to predicted labels
pred_labels = np.argmax(pred_prob, axis=1)

# Get class names (brand names)
class_names = list(val_data.class_indices.keys())

# Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()