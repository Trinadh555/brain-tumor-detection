"""
Brain tumor MRI classification with transfer learning (MobileNetV2, ImageNet weights).

Run from the folder that contains "Training" and "Testing":
    python train_brain_tumor_transfer.py

Why this should beat the small 64x64 CNN:
  * A pretrained backbone already knows edges, textures and shapes.
  * A larger input (160x160) keeps the detail that separates glioma from
    meningioma. The 64x64 CNN lost most of it.

It saves the SAME file names as train_brain_tumor.py (brain_tumor_model.keras,
class_names.json), so predict_brain_tumor.py and the app work unchanged.
Copy your old brain_tumor_model.keras somewhere first if you want to keep it.

Notes:
  * The first run downloads the ImageNet weights, so it needs internet.
  * On a CPU each epoch takes a few minutes. Google Colab with a free GPU
    runs this in a few minutes in total.
  * The Testing folder is used once, at the very end.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================
# 0. Settings
# =====================
SEED = 42
IMG_SIZE = 160          # MobileNetV2 accepts 96, 128, 160, 192, 224
BATCH_SIZE = 32
VAL_SPLIT = 0.15
HEAD_EPOCHS = 8         # phase 1: train only the new classifier head
FINE_TUNE_EPOCHS = 15   # phase 2: also fine-tune the last backbone layers
UNFREEZE_LAST = 40      # number of backbone layers to fine-tune
TRAIN_DIR = "Training"
TEST_DIR = "Testing"
MODEL_PATH = "brain_tumor_model.keras"

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =====================
# 1. Data
# =====================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=VAL_SPLIT,
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=VAL_SPLIT)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

flow_args = dict(
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)
training_set = train_datagen.flow_from_directory(
    TRAIN_DIR, subset="training", shuffle=True, seed=SEED, **flow_args
)
validation_set = val_datagen.flow_from_directory(
    TRAIN_DIR, subset="validation", shuffle=False, **flow_args
)
test_set = test_datagen.flow_from_directory(TEST_DIR, shuffle=False, **flow_args)

class_names = [n for n, _ in sorted(training_set.class_indices.items(), key=lambda kv: kv[1])]
print("Class order:", class_names)
assert class_names == list(test_set.class_indices.keys()), "Train/Test class folders differ"

with open("class_names.json", "w") as f:
    json.dump({"class_names": class_names, "img_size": IMG_SIZE}, f, indent=2)

# =====================
# 2. Model
# =====================
# Inputs stay in [0, 1] (same as the prediction code). The first layer maps
# them to [-1, 1], which is what MobileNetV2 expects.
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

# =====================
# 3. Phase 1: train the classifier head only
# =====================
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
h1 = model.fit(
    training_set,
    epochs=HEAD_EPOCHS,
    validation_data=validation_set,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
)

# =====================
# 4. Phase 2: fine-tune the last backbone layers at a low learning rate
# =====================
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False  # keep BatchNorm statistics stable

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(
    training_set,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_set,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ],
)

model.save(MODEL_PATH)

# =====================
# 5. Final evaluation on the untouched Testing folder
# =====================
test_loss, test_acc = model.evaluate(test_set, verbose=0)
print(f"\nTEST accuracy: {test_acc * 100:.2f}%   (loss {test_loss:.4f})")

y_pred = np.argmax(model.predict(test_set, verbose=0), axis=1)
y_true = test_set.classes
print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

# =====================
# 6. Plots
# =====================
acc = h1.history["accuracy"] + h2.history["accuracy"]
val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"] + h2.history["loss"]
val_loss = h1.history["val_loss"] + h2.history["val_loss"]
split = len(h1.history["loss"])

plt.figure(figsize=(10, 4))
for i, (a, b, title) in enumerate([(acc, val_acc, "Accuracy"), (loss, val_loss, "Loss")], start=1):
    plt.subplot(1, 2, i)
    plt.plot(a, label="Train")
    plt.plot(b, label="Validation")
    plt.axvline(split - 1, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)

plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

print(f"\nSaved: {MODEL_PATH}, class_names.json, training_curves.png, confusion_matrix.png")