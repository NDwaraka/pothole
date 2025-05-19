import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = (128, 128)
BATCH_SIZE = 3
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

dataset_path = './dataset'

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_data, epochs=10, validation_data=val_data)

model.save("pothole_model.h5")

def predict_pothole(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print(f"[ERROR] Could not read image from path: {img_path}")
        return

    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    label = "Pothole" if prediction >= 0.5 else "No Pothole"
    print(f"Prediction: {label} ({prediction:.2f})")

    cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if label == "No Pothole" else (0,0,255), 2)
    cv2.imshow("Prediction", cv2.resize(img, (300, 300)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_pothole('./dataset/pothole/download (2).jpg')
