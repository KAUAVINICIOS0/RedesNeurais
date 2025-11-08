# ===========================================================
# IMPORTS
# ===========================================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import seaborn as sns

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# ===========================================================
# LOAD DATASET
# ===========================================================
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# ===========================================================
# LABEL REMAPPING (URBAN = 0 | NATURE = 1)
# ===========================================================
urban  = [1, 9, 8]               
nature = [2, 3, 4, 5, 6, 7]      

def relabel(y):
    y_bin = np.zeros_like(y)
    for i, val in enumerate(y):
        y_bin[i] = 0 if val in urban else 1
    return y_bin

y_train = relabel(y_train)
y_test  = relabel(y_test)


# ===========================================================
# NORMALIZATION BEFORE SPLIT
# ===========================================================
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0


# ===========================================================
# TRAIN/VALID SPLIT
# ===========================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)


# ===========================================================
# MODEL
# ===========================================================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")  # binary
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ===========================================================
# TRAINING
# ===========================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=20,
    batch_size=32
)


# ===========================================================
# TEST EVALUATION
# ===========================================================
loss, acc = model.evaluate(X_test, y_test)
print("\n=== TEST RESULTS ===")
print("Accuracy:", acc)
print("Loss:", loss)


# ===========================================================
# CONFUSION MATRIX
# ===========================================================
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype("int")

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()


# ===========================================================
# CLASSIFICATION REPORT
# ===========================================================
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))


# ===========================================================
# ROC CURVE + AUC
# ===========================================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title(f"ROC Curve (AUC={auc_score:.3f})")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# ===========================================================
# TRAINING CURVES
# ===========================================================

# LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["train", "val"])
plt.title("Loss Over Epochs")
plt.show()

# ACCURACY
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["train", "val"])
plt.title("Accuracy Over Epochs")
plt.show()

# ===========================================================
# VISUALIZAR PREVISÕES EM IMAGENS
# ===========================================================
import random

label_names = {0: "URBANO", 1: "NATURAL"}

# Seleciona algumas imagens aleatórias do teste
indices = random.sample(range(len(X_test)), 12)

plt.figure(figsize=(12, 10))
for i, idx in enumerate(indices):
    img = X_test[idx]
    true_label = label_names[int(y_test[idx])]
    pred_label = label_names[int(y_pred[idx])]
    
    color = "green" if true_label == pred_label else "red"
    
    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"T:{true_label}\nP:{pred_label}", color=color)

plt.tight_layout()
plt.show()